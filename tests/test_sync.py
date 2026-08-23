import pytest
from subzero.sync import probe_audio_delay, auto_sync_file

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
