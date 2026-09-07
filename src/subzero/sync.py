"""Audio speech synchronization and alignment for subtitles."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import Tuple

from .extract import require_ffmpeg, ToolError
from .shift import shift_file, shift_timestamps


class SyncResult(tuple):
    target: Path
    count: int
    offset: float
    method: str
    report: dict | None

    def __new__(cls, target, count, offset, method="container_skew", report=None):
        inst = super().__new__(cls, (Path(target), int(count), float(offset)))
        inst.target = Path(target)
        inst.count = int(count)
        inst.offset = float(offset)
        inst.method = str(method)
        inst.report = report
        return inst


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
    cache_dir: str | Path | None = None,
) -> SyncResult:
    """Automatically synchronize subtitle to spoken dialogue using speech VAD,
    falling back to container PTS audio delay if speech alignment is unavailable."""
    require_ffmpeg()
    video_p = Path(video_path)
    sub_p = Path(subtitle_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Video file not found: {video_p}")
    if not sub_p.exists():
        raise FileNotFoundError(f"Subtitle file not found: {sub_p}")

    try:
        from .reference import build_reference, verify_text
        from .timing import correction

        cache = Path(cache_dir) if cache_dir else (Path.home() / ".cache/subzero/references")
        ref = build_reference(video_p, cache)
        content = sub_p.read_text(encoding="utf-8", errors="replace")
        report = verify_text(content, ref)

        if report.status == "pass":
            target = Path(output) if output else sub_p
            return SyncResult(target, 0, 0.0, method="speech_aligned", report=report.json())

        change = correction(report)
        if change is not None:
            scale, offset = change
            shifted_text, count = shift_timestamps(content, delta_seconds=offset, scale_factor=scale)
            repaired_report = verify_text(shifted_text, ref)
            if repaired_report.status == "pass":
                target = Path(output) if output else sub_p
                if not dry:
                    if backup_dir and target == sub_p:
                        bdir = Path(backup_dir)
                        bdir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(sub_p, bdir / sub_p.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(shifted_text, encoding="utf-8")
                return SyncResult(target, count, offset, method="speech_synced", report=repaired_report.json())
    except Exception:
        pass

    offset = probe_audio_delay(video_p)
    target, count = shift_file(
        sub_p,
        delta_seconds=offset,
        output=output,
        backup_dir=backup_dir,
        dry=dry,
    )
    return SyncResult(target, count, offset, method="container_skew", report=None)
