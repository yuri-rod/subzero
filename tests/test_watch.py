import json
import time

from subzero import Options
from subzero.watch import Watcher

CUE = "1\n00:00:01,000 --> 00:00:02,000\n[music]\n\n" \
      "2\n00:00:03,000 --> 00:00:04,000\nReal line.\n\n"


def write(path, text="", age=0):
    path.write_text(text or CUE, encoding="utf-8")
    if age:
        old = time.time() - age
        import os
        os.utime(path, (old, old))
    return path


def watcher(tmp_path, **kw):
    kw.setdefault("state_path", tmp_path / "state.json")
    kw.setdefault("settle", 0)
    return Watcher(roots=[tmp_path], opts=Options(), **kw)


def test_sweep_fixes_a_file(tmp_path):
    f = write(tmp_path / "a.srt")
    assert watcher(tmp_path).sweep() == (1, 0)
    assert "[music]" not in f.read_text(encoding="utf-8")


def test_second_sweep_does_nothing(tmp_path):
    write(tmp_path / "a.srt")
    w = watcher(tmp_path)
    w.sweep()
    assert w.sweep() == (0, 0)


def test_state_survives_a_new_watcher(tmp_path):
    write(tmp_path / "a.srt")
    watcher(tmp_path).sweep()
    assert watcher(tmp_path).sweep() == (0, 0)


def test_state_records_size_not_just_mtime(tmp_path):
    """On mtime alone the tool's own rewrite reads as a fresh change forever."""
    write(tmp_path / "a.srt")
    w = watcher(tmp_path)
    w.sweep()
    stamp = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert all(len(v) == 2 for v in stamp.values())


def test_a_changed_file_is_picked_up_again(tmp_path):
    f = write(tmp_path / "a.srt")
    w = watcher(tmp_path)
    w.sweep()
    write(f)
    assert w.sweep() == (1, 0)


def test_recent_files_are_left_alone(tmp_path):
    write(tmp_path / "a.srt")
    assert watcher(tmp_path, settle=600).sweep() == (0, 0)


def test_skip_glob_is_honoured(tmp_path):
    write(tmp_path / "a.en.srt")
    assert watcher(tmp_path, skip="*.en.srt").sweep() == (0, 0)


def test_backup_keeps_the_first_original(tmp_path):
    f = write(tmp_path / "a.srt")
    backup = tmp_path / "bak"
    w = watcher(tmp_path, backup_dir=str(backup))
    w.sweep()
    assert "[music]" in (backup / "a.srt").read_text(encoding="utf-8")


def test_unreadable_file_counts_as_failure_not_crash(tmp_path):
    (tmp_path / "bad.srt").write_bytes(b"not a subtitle at all")
    assert watcher(tmp_path).sweep() == (0, 1)


def test_two_edits_in_the_same_second_are_both_seen(tmp_path):
    """Truncating mtime to whole seconds hid an edit that landed in the same
    second as the sweep that recorded it."""
    f = write(tmp_path / "a.srt")
    w = watcher(tmp_path)
    w.sweep()
    write(f)
    stamp = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    recorded = next(iter(stamp.values()))[0]
    assert w.sweep() == (1, 0)
    assert isinstance(recorded, float)
