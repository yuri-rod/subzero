"""Timestamp shifting utility for subtitles."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _to_timestamp(value: float, sep: str = ",") -> str:
    value = max(0.0, value)
    ms = int(round(value * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def shift_timestamps(text: str, delta_seconds: float) -> tuple[str, int]:
    """Shift all subtitle timecodes by delta_seconds.
    
    Returns (shifted_text, count_of_shifted_cues).
    """
    shifted_count = 0

    def repl(m: re.Match) -> str:
        nonlocal shifted_count
        g = m.groups()
        sep = "," if "," in m.group(0) else "."
        start_sec = _to_seconds(*g[:4]) + delta_seconds
        end_sec = _to_seconds(*g[4:]) + delta_seconds
        shifted_count += 1
        return f"{_to_timestamp(start_sec, sep)} --> {_to_timestamp(end_sec, sep)}"

    return TIME.sub(repl, text), shifted_count


def shift_file(
    path: Path,
    delta_seconds: float,
    output: Path | None = None,
    backup_dir: str | Path | None = None,
    dry: bool = False,
) -> tuple[Path, int]:
    """Shift a subtitle file in place or to a target output path."""
    content = path.read_text(encoding="utf-8", errors="replace")
    shifted_text, count = shift_timestamps(content, delta_seconds)

    target = Path(output) if output else path

    if dry:
        return target, count

    if backup_dir and target == path:
        bdir = Path(backup_dir)
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bdir / path.name)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(shifted_text, encoding="utf-8")
    return target, count
