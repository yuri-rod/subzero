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


@pytest.mark.parametrize("dry", [False, True])
def test_auto_sync_file_speech_already_aligned(monkeypatch, tmp_path, dry):
    video = tmp_path / 'movie.mkv'
    video.write_bytes(b'video')
    sub = tmp_path / 'movie.srt'
    sub.write_bytes(b'1\r\n00:00:01,000 --> 00:00:03,000\r\nHello world\r\n')
    monkeypatch.setattr('subzero.sync.require_ffmpeg', lambda: None)

    write_text = Path.write_text
    monkeypatch.setattr(Path, "write_text",
                        lambda path, text, **kwargs: write_text(path, text, newline="\r\n", **kwargs))

    fake_report = Report(status='pass', reason='Timing agrees across the dialogue span')
    monkeypatch.setattr('subzero.reference.build_reference', lambda vid, cache: {})
    monkeypatch.setattr('subzero.reference.verify_text', lambda txt, ref: fake_report)

    output = tmp_path / "synced" / "movie.srt"
    res = auto_sync_file(video, sub, output=output, dry=dry)
    assert res.method == 'speech_aligned'
    assert res.offset == 0.0
    assert res.report['status'] == 'pass'
    assert output.exists() is not dry
    if not dry:
        assert output.read_bytes() == sub.read_bytes()


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


def test_sync_output_error_does_not_fall_back_to_container_delay(monkeypatch, tmp_path):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"video")
    subtitle = tmp_path / "movie.srt"
    original = "1\n00:00:01,000 --> 00:00:03,000\nHello.\n"
    subtitle.write_text(original, encoding="utf-8")
    monkeypatch.setattr("subzero.sync.require_ffmpeg", lambda: None)
    monkeypatch.setattr("subzero.reference.build_reference", lambda *args: {})
    monkeypatch.setattr("subzero.reference.verify_text", lambda *args: Report("pass", "Aligned"))

    def refuse_write(*args, **kwargs):
        raise PermissionError("Output is read-only")

    def unexpected_fallback(*args):
        pytest.fail("A failed write must not change the synchronization method")

    monkeypatch.setattr(Path, "write_bytes", refuse_write)
    monkeypatch.setattr("subzero.sync.probe_audio_delay", unexpected_fallback)
    with pytest.raises(PermissionError, match="Output is read-only"):
        auto_sync_file(video, subtitle, output=tmp_path / "synced.srt")
    assert subtitle.read_text(encoding="utf-8") == original


def test_auto_sync_preserves_first_backup_and_rejects_symlink(monkeypatch, tmp_path):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"video")
    sub = tmp_path / "movie.srt"
    original = "1\n00:00:05,000 --> 00:00:07,000\nDelayed cue\n"
    sub.write_text(original, encoding="utf-8")
    backup_dir = tmp_path / "backups"
    monkeypatch.setattr("subzero.sync.require_ffmpeg", lambda: None)
    monkeypatch.setattr("subzero.reference.build_reference", lambda vid, cache: {})
    monkeypatch.setattr(
        "subzero.reference.verify_text",
        lambda txt, ref: Report("pass", "ok") if "00:00:03,000" in txt or "00:00:01,000" in txt else Report("reject", "drift"),
    )
    monkeypatch.setattr("subzero.timing.correction", lambda rep: (1.0, -2.0))

    auto_sync_file(video, sub, backup_dir=backup_dir)
    sub.write_text("1\n00:00:06,000 --> 00:00:08,000\nSecond shift\n", encoding="utf-8")
    monkeypatch.setattr(
        "subzero.reference.verify_text",
        lambda txt, ref: Report("pass", "ok") if "00:00:04,000" in txt else Report("reject", "drift"),
    )
    auto_sync_file(video, sub, backup_dir=backup_dir)

    assert (backup_dir / "movie.srt").read_text(encoding="utf-8") == original

    link = tmp_path / "link.srt"
    try:
        link.symlink_to(sub)
    except OSError:
        pytest.skip("Creating symlinks requires platform permissions")
    with pytest.raises(OSError, match="symbolic link"):
        auto_sync_file(video, link)


def test_auto_sync_file_piecewise_synced_after_segment_correction(monkeypatch, tmp_path):
    from subzero.timing import Segment

    video = tmp_path / 'movie.mkv'
    video.write_bytes(b'video')
    sub = tmp_path / 'movie.srt'
    sub.write_text(
        '1\n00:01:03,000 --> 00:01:05,000\nAct 1 cue\n\n'
        '2\n00:15:33,000 --> 00:15:35,000\nAct 2 cue\n',
        encoding='utf-8',
    )
    monkeypatch.setattr('subzero.sync.require_ffmpeg', lambda: None)

    initial_report = Report(status='reject', reason='Severe timing offset')
    repaired_report = Report(status='pass', reason='Timing agrees after piecewise shift')

    reports = [initial_report, repaired_report]

    def fake_verify(txt, ref):
        return reports.pop(0)

    monkeypatch.setattr('subzero.reference.build_reference', lambda vid, cache: {})
    monkeypatch.setattr('subzero.reference.verify_text', fake_verify)
    monkeypatch.setattr('subzero.timing.correction', lambda rep: None)
    monkeypatch.setattr('subzero.timing.piecewise_correction', lambda rep: [
        Segment(start=0.0, end=600.0, offset=-3.0),
        Segment(start=600.0, end=1800.0, offset=-33.0),
    ])

    res = auto_sync_file(video, sub)
    assert res.method == 'piecewise_synced'
    assert abs(res.offset - (-3.0)) < 0.001
    assert res.count == 2
    assert res.report['status'] == 'pass'
    content = sub.read_text(encoding='utf-8')
    assert '00:01:00,000 --> 00:01:02,000' in content
    assert '00:15:00,000 --> 00:15:02,000' in content
