"""Audio speech synchronization and alignment for subtitles."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Tuple

from .extract import require_ffmpeg, ToolError
from .shift import shift_file, shift_timestamps


def probe_audio_delay(video_path: str | Path) -> float:
    """Probe container audio-to-video start time skew using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time",
        "-of", "json", str(video_path)
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as err:
        raise ToolError(f"ffprobe execution failed: {err}") from err

    if getattr(out, "returncode", 1) != 0:
        return 0.0

    try:
        streams = json.loads(out.stdout or "{}").get("streams", [])
    except ValueError:
        return 0.0

    audio_start = 0.0
    video_start = 0.0
    has_audio = has_video = False

    for st in streams:
        ctype = st.get("codec_type")
        stime = st.get("start_time")
        if stime is not None:
            try:
                val = float(stime)
                if ctype == "audio" and not has_audio:
                    audio_start = val
                    has_audio = True
                elif ctype == "video" and not has_video:
                    video_start = val
                    has_video = True
            except (ValueError, TypeError):
                pass

    if has_audio and has_video:
        skew = audio_start - video_start
        return skew if abs(skew) >= 0.05 else 0.0
    return 0.0


def auto_sync_file(
    video_path: str | Path,
    subtitle_path: str | Path,
    output: str | Path | None = None,
    backup_dir: str | Path | None = None,
    dry: bool = False,
) -> Tuple[Path, int, float]:
    """Automatically detect container skew and shift subtitle file to match."""
    require_ffmpeg()
    video_p = Path(video_path)
    sub_p = Path(subtitle_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Video file not found: {video_p}")
    if not sub_p.exists():
        raise FileNotFoundError(f"Subtitle file not found: {sub_p}")

    offset = probe_audio_delay(video_p)
    target, count = shift_file(
        sub_p,
        delta_seconds=offset,
        output=output,
        backup_dir=backup_dir,
        dry=dry,
    )
    return target, count, offset
