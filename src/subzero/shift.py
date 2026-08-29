"""Timestamp shifting and timecode stretching utility for subtitles."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

COMMON_FPS = {
    "23.976": 23.976,
    "23.98": 23.976,
    "24": 24.0,
    "25": 25.0,
    "29.97": 29.97,
    "30": 30.0,
    "50": 50.0,
    "59.94": 59.94,
    "60": 60.0,
}


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _to_timestamp(value: float, sep: str = ",") -> str:
    value = max(0.0, value)
    ms = int(round(value * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def calculate_fps_factor(from_fps: float | str, to_fps: float | str) -> float:
    """Calculate speed ratio when converting subtitle framerate (e.g. 23.976 -> 25)."""
    f1 = COMMON_FPS.get(str(from_fps).strip(), float(from_fps))
    f2 = COMMON_FPS.get(str(to_fps).strip(), float(to_fps))
    if f1 <= 0 or f2 <= 0:
        raise ValueError(f"Invalid frame rate: {from_fps} -> {to_fps}")
    return f1 / f2


def shift_timestamps(
    text: str,
    delta_seconds: float = 0.0,
    scale_factor: float = 1.0,
) -> tuple[str, int]:
    """Shift and/or stretch all subtitle timecodes.
    
    Returns (processed_text, count_of_shifted_cues).
    """
    shifted_count = 0

    def repl(m: re.Match) -> str:
        nonlocal shifted_count
        g = m.groups()
        sep = "," if "," in m.group(0) else "."
        start_sec = max(0.0, (_to_seconds(*g[:4]) * scale_factor) + delta_seconds)
        end_sec = max(start_sec, (_to_seconds(*g[4:]) * scale_factor) + delta_seconds)
        shifted_count += 1
        return f"{_to_timestamp(start_sec, sep)} --> {_to_timestamp(end_sec, sep)}"

    return TIME.sub(repl, text), shifted_count


def shift_file(
    path: Path,
    delta_seconds: float = 0.0,
    scale_factor: float = 1.0,
    from_fps: float | str | None = None,
    to_fps: float | str | None = None,
    output: Path | None = None,
    backup_dir: str | Path | None = None,
    dry: bool = False,
) -> tuple[Path, int]:
    """Shift and/or speed-stretch a subtitle file in place or to a target output path."""
    if from_fps is not None and to_fps is not None:
        scale_factor = scale_factor * calculate_fps_factor(from_fps, to_fps)

    content = path.read_text(encoding="utf-8", errors="replace")
    shifted_text, count = shift_timestamps(content, delta_seconds=delta_seconds, scale_factor=scale_factor)

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
