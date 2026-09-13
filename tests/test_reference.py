from subzero import reference
import json
import subprocess


def test_fingerprint_changes_when_video_is_replaced(tmp_path):
    video = tmp_path/'movie.mkv'
    video.write_bytes(b'one')
    first = reference.fingerprint(video)
    video.write_bytes(b'two')
    assert reference.fingerprint(video) != first


def test_forced_and_commentary_tracks_are_excluded():
    tracks = [
        {'index':1,'codec_type':'subtitle','codec_name':'subrip','disposition':{'forced':1}},
        {'index':2,'codec_type':'subtitle','codec_name':'subrip','tags':{'title':'Commentary'}},
        {'index':3,'codec_type':'subtitle','codec_name':'subrip','tags':{'language':'eng'}},
    ]
    assert [s['index'] for s in reference.dialogue_tracks(tracks)] == [3]


def test_absent_audio_evidence_cannot_certify_a_subtitle():
    report = reference.verify_text('1\n00:00:01,000 --> 00:00:02,000\nHello\n',
                                   {'speech':[], 'text':'', 'spans':[]})
    assert report.status == 'inconclusive'


def test_cached_reference_recovers_duration_without_reextracting_audio(tmp_path, monkeypatch):
    video = tmp_path/'movie.mkv'
    video.write_bytes(b'video')
    cache = tmp_path/f'{reference.fingerprint(video)}.json'
    cache.write_text(json.dumps({'speech': [[1, 2]], 'text': '', 'spans': []}))

    def probe(argv, **kwargs):
        if argv[0] != 'ffprobe':
            raise AssertionError('Cached audio should not be extracted again')
        return subprocess.CompletedProcess(argv, 0, b'{"format":{"duration":"5150.976"}}')

    monkeypatch.setattr(reference, '_run', probe)
    cached = reference.build_reference(video, tmp_path)
    assert cached['duration'] == 5150.976
    assert cached['speech'] == [[1, 2]]


def test_reference_closes_cache_file_before_replacement(tmp_path, monkeypatch):
    from pathlib import Path
    import faster_whisper.vad

    video = tmp_path / 'movie.mkv'
    video.write_bytes(b'video')
    create = reference.tempfile.NamedTemporaryFile
    replace = reference.os.replace
    handles = []

    def stage(*args, **kwargs):
        handle = create(*args, **kwargs)
        handles.append(handle)
        return handle

    def publish(source, target):
        assert all(handle.closed for handle in handles if Path(handle.name) == Path(source))
        return replace(source, target)

    def extract(argv, **kwargs):
        if argv[0] == 'ffprobe':
            return subprocess.CompletedProcess(argv, 0, json.dumps({
                'format': {'duration': 180},
                'streams': [{'index': 0, 'codec_type': 'audio'}],
            }).encode())
        kwargs['stdout'].write(b'\x00' * 64)
        return subprocess.CompletedProcess(argv, 0, b'')

    monkeypatch.setattr(reference.tempfile, 'NamedTemporaryFile', stage)
    monkeypatch.setattr(reference.os, 'replace', publish)
    monkeypatch.setattr(reference, '_run', extract)
    monkeypatch.setattr(faster_whisper.vad, 'get_speech_timestamps', lambda *a: [])
    built = reference.build_reference(video, tmp_path / 'cache')
    cached = next((tmp_path / 'cache').glob('*.json'))
    assert json.loads(cached.read_text()) == built
    assert all(handle.closed for handle in handles)
    assert list(cached.parent.iterdir()) == [cached]
