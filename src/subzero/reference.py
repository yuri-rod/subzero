from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .timing import VERSION, Report, evaluate, spans


def fingerprint(video):
    path = Path(video).resolve(strict=True)
    stat = path.stat()
    digest = hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}:{VERSION}'.encode())
    with path.open('rb') as handle:
        digest.update(handle.read(65536))
        handle.seek(max(0,stat.st_size-65536))
        digest.update(handle.read())
    return digest.hexdigest()


def dialogue_tracks(streams):
    selected = []
    for stream in streams:
        title = stream.get('tags',{}).get('title','').lower()
        if (stream.get('codec_type') == 'subtitle'
                and stream.get('codec_name') in ('subrip','ass','ssa','mov_text','webvtt')
                and not stream.get('disposition',{}).get('forced')
                and not any(word in title for word in ('forced','signs','commentary'))):
            selected.append(stream)
    return sorted(selected,key=lambda s: s.get('tags',{}).get('language') not in ('eng','en'))


def _run(argv, **kwargs):
    proc = subprocess.run(argv, stderr=subprocess.PIPE, timeout=1800, **kwargs)
    if proc.returncode:
        raise RuntimeError(f'{argv[0]} failed: {proc.stderr.decode(errors="replace")[-300:]}')
    return proc


def build_reference(video, cache_dir):
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    key = fingerprint(video)
    cache = Path(cache_dir)/f'{key}.json'
    if cache.exists():
        reference = json.loads(cache.read_text())
        if 'duration' not in reference:
            info = json.loads(_run(['ffprobe','-v','error','-show_entries','format=duration',
                                   '-of','json',str(video)],stdout=subprocess.PIPE).stdout)
            duration = float(info.get('format',{}).get('duration') or 0)
            if not 120 <= duration <= 14400:
                raise ValueError('Audio verification supports videos from two minutes to four hours')
            reference['duration'] = duration
        return reference
    video = str(Path(video).resolve(strict=True))
    info = json.loads(_run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',video],
                           stdout=subprocess.PIPE).stdout)
    duration = float(info.get('format',{}).get('duration') or 0)
    if not 120 <= duration <= 14400:
        raise ValueError('Audio verification supports videos from two minutes to four hours')
    streams = info.get('streams',[])
    audio = next((s for s in streams if s.get('codec_type') == 'audio'), None)
    if audio is None:
        raise ValueError('Video has no audio stream')
    speech = []
    with tempfile.TemporaryFile() as pcm:
        _run(['ffmpeg','-v','error','-nostdin','-i',video,'-map',f'0:{audio["index"]}',
              '-vn','-ac','1','-ar','16000','-af','aresample=async=1:first_pts=0',
              '-f','f32le','pipe:1'],stdout=pcm)
        samples = pcm.tell()//4
        if samples == 0:
            raise ValueError('Audio extraction produced no samples')
        wave = np.memmap(pcm,dtype=np.float32,mode='r',shape=(samples,))
        block = 16000*300
        opts = VadOptions(min_speech_duration_ms=200,min_silence_duration_ms=200,speech_pad_ms=100)
        for start in range(0,samples,block):
            for chunk in get_speech_timestamps(wave[start:start+block],opts):
                speech.append(((start+chunk['start'])/16000,(start+chunk['end'])/16000))
        del wave
    reference = {'fingerprint':key,'duration':duration,'speech':speech,'spans':[], 'text':'',
                 'audio_index':audio['index'],'language':audio.get('tags',{}).get('language','und')}
    for stream in dialogue_tracks(streams):
        text = _run(['ffmpeg','-v','error','-nostdin','-i',video,'-map',f'0:{stream["index"]}',
                     '-f','srt','pipe:1'],stdout=subprocess.PIPE).stdout.decode('utf-8',errors='replace')
        track = spans(text)
        if len(track) < 20:
            continue
        audio_check = evaluate(track,speech,tolerance=1.0,search_seconds=20)
        is_pass = audio_check.status == 'pass'
        if not is_pass and audio_check.status == 'inconclusive' and audio_check.reason == 'Isolated timing mismatch needs review':
            conf = [w for w in audio_check.windows if w.confident]
            if len(conf) >= 15 and all(abs(w.offset) <= 5.0 for w in conf):
                is_pass = True
        if is_pass:
            reference.update(text=text,spans=track,subtitle_index=stream['index'],
                             language=stream.get('tags',{}).get('language','und'))
            break
    if fingerprint(video) != key:
        raise RuntimeError('Video changed during reference extraction')
    cache.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=cache.parent,delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(reference,handle)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(tmp,cache)
        finally:
            handle.close()
            tmp.unlink(missing_ok=True)
    return reference


def verify_text(text, reference, phase=0):
    candidate = spans(text)
    speech = reference.get('speech',[])
    if not speech:
        return Report('inconclusive','No independent audio evidence')
    has_spans = bool(reference.get('spans'))
    track = reference['spans'] if has_spans else speech
    search_seconds = 300 if has_spans else 20
    tolerance = 0.5 if has_spans else 1.0
    report = evaluate(candidate, track, tolerance=tolerance, search_seconds=search_seconds, phase=phase)
    if report.status == 'inconclusive' and report.reason == 'Isolated timing mismatch needs review':
        conf = [w for w in report.windows if w.confident]
        if len(conf) >= 15 and all(abs(w.offset) <= 5.0 for w in conf):
            report = Report('pass', 'Timing agrees across the dialogue span', report.windows)
    if report.status != 'pass':
        return report
    if has_spans:
        audio = evaluate(candidate, speech, tolerance=1.0, search_seconds=20, phase=phase)
        if audio.status == 'inconclusive' and audio.reason == 'Isolated timing mismatch needs review':
            conf = [w for w in audio.windows if w.confident]
            if len(conf) >= 15 and all(abs(w.offset) <= 5.0 for w in conf):
                audio = Report('pass', 'Audio confirmed with isolated outlier tolerated', audio.windows)
        if audio.status != 'pass':
            return Report('inconclusive', 'Audio evidence does not confirm the subtitle', report.windows)
    return report
