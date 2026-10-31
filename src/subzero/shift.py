"""Timestamp shifting and timecode stretching utility for subtitles."""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from .core import read

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
    if not math.isfinite(f1) or not math.isfinite(f2) or f1 <= 0 or f2 <= 0:
        raise ValueError(f"Invalid frame rate: {from_fps} -> {to_fps}")
    return f1 / f2


def shift_timestamps(
    text: str,
    delta_seconds: float = 0.0,
    scale_factor: float = 1.0,
    segments: list | None = None,
) -> tuple[str, int]:
    """Shift and/or stretch subtitle timecodes.

    Supports uniform global shift/scale or piecewise segment shifts across
    commercial breaks and segmented masters.
    """
    if not math.isfinite(delta_seconds):
        raise ValueError("Shift seconds must be finite")
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("Scale factor must be finite and positive")
    shifted_count = 0
    last_end = 0.0

    def repl(m: re.Match) -> str:
        nonlocal shifted_count, last_end
        g = m.groups()
        sep = "," if "," in m.group(0) else "."
        raw_start = _to_seconds(*g[:4])
        raw_end = _to_seconds(*g[4:])

        if segments:
            chosen = segments[-1]
            for seg in segments:
                s_start = seg.start if hasattr(seg, "start") else seg[0]
                s_end = seg.end if hasattr(seg, "end") else seg[1]
                if s_start <= raw_start < s_end:
                    chosen = seg
                    break
            offset = chosen.offset if hasattr(chosen, "offset") else chosen[2]
            scale = getattr(chosen, "scale", 1.0) if hasattr(chosen, "scale") else (chosen[3] if len(chosen) > 3 else 1.0)
        else:
            offset = delta_seconds
            scale = scale_factor

        start_sec = max(0.0, (raw_start * scale) + offset)
        end_sec = max(start_sec, (raw_end * scale) + offset)
        if segments and start_sec < last_end:
            start_sec = last_end + 0.05
            end_sec = max(start_sec + 0.1, end_sec)
        last_end = end_sec
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
    segments: list | None = None,
) -> tuple[Path, int]:
    """Shift and/or speed-stretch a subtitle file in place or to a target output path."""
    if (from_fps is None) != (to_fps is None):
        raise ValueError("Source and target frame rates must be supplied together")
    if from_fps is not None and to_fps is not None:
        scale_factor = scale_factor * calculate_fps_factor(from_fps, to_fps)

    path = Path(path)
    target = Path(output) if output else path
    if path.is_symlink() or target.is_symlink():
        raise OSError(f"Refusing to modify symbolic link: {target if target.is_symlink() else path}")

    content, _ = read(path)
    shifted_text, count = shift_timestamps(
        content,
        delta_seconds=delta_seconds,
        scale_factor=scale_factor,
        segments=segments,
    )

    if dry:
        return target, count

    if backup_dir and target == path:
        bdir = Path(backup_dir)
        bdir.mkdir(parents=True, exist_ok=True)
        bfile = bdir / path.name
        if bfile.is_symlink():
            raise OSError(f"Refusing to write backup through symbolic link: {bfile}")
        if not bfile.exists():
            shutil.copy2(path, bfile)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=target.parent, prefix=".subzero-", suffix=".subzero-part",
    ) as fh:
        try:
            fh.write(shifted_text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
            if path.exists():
                shutil.copymode(path, fh.name)
            os.replace(fh.name, target)
        finally:
            fh.close()
            if os.path.exists(fh.name):
                os.unlink(fh.name)
    return target, count
