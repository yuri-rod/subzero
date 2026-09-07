from pathlib import Path
import pytest

from subzero.sync import SyncResult, auto_sync_file, probe_audio_delay
from subzero.timing import Report


def test_probe_audio_delay_mock(monkeypatch, tmp_path):
    p = tmp_path / 'video.mp4'
    p.write_bytes(b'x')
    import subprocess
    class FakeOut:
        returncode = 0
        stdout = '{"streams": [{"codec_type": "video", "start_time": "0.000"}, {"codec_type": "audio", "start_time": "1.500"}]}'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: FakeOut())
    delay = probe_audio_delay(p)
    assert abs(delay - 1.5) < 0.001


def test_sync_result_unpacking_and_metadata():
    p = Path('/tmp/test.srt')
    res = SyncResult(p, 42, 1.25, method='speech_synced', report={'status': 'pass'})
    target, count, offset = res
    assert target == p
    assert count == 42
    assert offset == 1.25
    assert res.target == p
    assert res.count == 42
    assert res.offset == 1.25
    assert res.method == 'speech_synced'
    assert res.report == {'status': 'pass'}


def test_auto_sync_file_fallback_to_container_delay(monkeypatch, tmp_path):
    video = tmp_path / 'movie.mkv'
    video.write_bytes(b'video')
    sub = tmp_path / 'movie.srt'
    sub.write_text('1\n00:00:01,000 --> 00:00:03,000\nHello world\n', encoding='utf-8')
    monkeypatch.setattr('subzero.sync.require_ffmpeg', lambda: None)
    monkeypatch.setattr('subzero.sync.probe_audio_delay', lambda p: 1.5)

    res = auto_sync_file(video, sub)
    assert res.method == 'container_skew'
    assert abs(res.offset - 1.5) < 0.001
    assert res.count == 1
    content = sub.read_text(encoding='utf-8')
    assert '00:00:02,500 --> 00:00:04,500' in content


def test_auto_sync_file_speech_already_aligned(monkeypatch, tmp_path):
    video = tmp_path / 'movie.mkv'
    video.write_bytes(b'video')
    sub = tmp_path / 'movie.srt'
    sub.write_text('1\n00:00:01,000 --> 00:00:03,000\nHello world\n', encoding='utf-8')
    monkeypatch.setattr('subzero.sync.require_ffmpeg', lambda: None)

    fake_report = Report(status='pass', reason='Timing agrees across the dialogue span')
    monkeypatch.setattr('subzero.reference.build_reference', lambda vid, cache: {})
    monkeypatch.setattr('subzero.reference.verify_text', lambda txt, ref: fake_report)

    res = auto_sync_file(video, sub)
    assert res.method == 'speech_aligned'
    assert res.offset == 0.0
    assert res.report['status'] == 'pass'


def test_auto_sync_file_speech_synced_after_correction(monkeypatch, tmp_path):
    video = tmp_path / 'movie.mkv'
    video.write_bytes(b'video')
    sub = tmp_path / 'movie.srt'
    sub.write_text('1\n00:00:05,000 --> 00:00:07,000\nDelayed cue\n', encoding='utf-8')
    monkeypatch.setattr('subzero.sync.require_ffmpeg', lambda: None)

    initial_report = Report(status='reject', reason='Delay drift')
    repaired_report = Report(status='pass', reason='Timing agrees after shift')

    reports = [initial_report, repaired_report]

    def fake_verify(txt, ref):
        return reports.pop(0)

    monkeypatch.setattr('subzero.reference.build_reference', lambda vid, cache: {})
    monkeypatch.setattr('subzero.reference.verify_text', fake_verify)
    monkeypatch.setattr('subzero.timing.correction', lambda rep: (1.0, -2.0))

    res = auto_sync_file(video, sub)
    assert res.method == 'speech_synced'
    assert abs(res.offset - (-2.0)) < 0.001
    assert res.count == 1
    assert res.report['status'] == 'pass'
    content = sub.read_text(encoding='utf-8')
    assert '00:00:03,000 --> 00:00:05,000' in content

