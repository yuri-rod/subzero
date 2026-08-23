"""Extract subtitle streams from video containers via ffmpeg/ffprobe.

Works with any container ffmpeg understands: mp4, mkv, mov, webm, avi, m4v,
mpeg-ts, m2ts, mlv (when the build supports it), and more. Soft-sub streams
are copied or converted; hard-burned captions cannot be recovered.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .convert import FORMATS, convert_file, convert_text, detect_format

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".m4v",
    ".ts",
    ".m2ts",
    ".mts",
    ".wmv",
    ".flv",
    ".mpg",
    ".mpeg",
    ".mlv",
    ".m2v",
    ".3gp",
    ".ogv",
    ".divx",
)

CODEC_TO_FORMAT = {
    "subrip": "srt",
    "srt": "srt",
    "ass": "ass",
    "ssa": "ssa",
    "webvtt": "vtt",
    "mov_text": "srt",
    "tx3g": "srt",
    "text": "srt",
    "hdmv_pgs_subtitle": "sup",
    "dvd_subtitle": "sub",
    "dvdsub": "sub",
    "pgssub": "sup",
    "xsub": "sub",
}

TEXT_CODECS = {
    "subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "tx3g", "text",
}


class ToolError(RuntimeError):
    """ffmpeg/ffprobe missing or failed."""


@dataclass(frozen=True)
class SubtitleStream:
    index: int
    codec: str
    language: str = ""
    title: str = ""
    is_text: bool = True
    disposition: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        bits = [f"#{self.index}", self.codec or "unknown"]
        if self.language:
            bits.append(self.language)
        if self.title:
            bits.append(self.title)
        bits.append("text" if self.is_text else "image")
        return " ".join(bits)


@dataclass
class ExtractResult:
    source: str
    outputs: list[str] = field(default_factory=list)
    streams: list[SubtitleStream] = field(default_factory=list)
    converted: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    message: str = ""


def which_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ToolError(
            f"{name} not found on PATH. Install ffmpeg "
            f"(https://ffmpeg.org) to extract subtitles from video."
        )
    return path


def require_ffmpeg() -> tuple[str, str]:
    return which_tool("ffmpeg"), which_tool("ffprobe")


def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise ToolError(str(e)) from e


def list_subtitle_streams(path) -> list[SubtitleStream]:
    """Return subtitle streams in *path* using ffprobe."""
    _, ffprobe = require_ffmpeg()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "s",
        "-show_entries",
        "stream=index,codec_name,codec_type:stream_tags=language,title:"
        "stream_disposition=default,forced,hearing_impaired",
        "-of", "json",
        str(path),
    ]
    proc = _run(cmd, timeout=120)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "ffprobe failed").strip()
        raise ToolError(err[:300])
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise ToolError(f"ffprobe returned invalid JSON: {e}") from e
    out: list[SubtitleStream] = []
    for s in data.get("streams") or []:
        tags = s.get("tags") or {}
        codec = (s.get("codec_name") or "").lower()
        out.append(
            SubtitleStream(
                index=int(s["index"]),
                codec=codec,
                language=str(tags.get("language") or tags.get("LANGUAGE") or ""),
                title=str(tags.get("title") or tags.get("TITLE") or ""),
                is_text=codec in TEXT_CODECS,
                disposition=dict(s.get("disposition") or {}),
            )
        )
    return out


def _stream_selector(stream: SubtitleStream, streams: list[SubtitleStream]) -> str:
    for n, s in enumerate(streams):
        if s.index == stream.index:
            return f"0:s:{n}"
    return f"0:{stream.index}"


def _default_output(path: Path, stream: SubtitleStream, fmt: str, out_dir: Path) -> Path:
    lang = stream.language or "und"
    return out_dir / f"{path.stem}.{lang}.{stream.index}.{fmt}"


def extract_stream(
    path,
    stream: SubtitleStream,
    streams: list[SubtitleStream] | None = None,
    output=None,
    fmt: str | None = None,
    out_dir=None,
    dry: bool = False,
) -> Path:
    """Extract one subtitle stream to a file. Returns the output path."""
    ffmpeg, _ = require_ffmpeg()
    path = Path(path)
    streams = streams if streams is not None else list_subtitle_streams(path)
    fmt = (fmt or CODEC_TO_FORMAT.get(stream.codec, "srt")).lower().lstrip(".")
    out_dir = Path(out_dir) if out_dir else path.parent
    out = Path(output) if output else _default_output(path, stream, fmt, out_dir)

    if dry:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    selector = _stream_selector(stream, streams)

    if not stream.is_text:
        cmd = [
            ffmpeg, "-y", "-i", str(path),
            "-map", selector, "-c", "copy", str(out),
        ]
        proc = _run(cmd)
        if proc.returncode != 0:
            raise ToolError((proc.stderr or "ffmpeg extract failed")[-400:])
        return out

    native = CODEC_TO_FORMAT.get(stream.codec, "srt")
    if native not in FORMATS:
        native = "srt"
    intermediate_fmt = native if fmt in FORMATS else fmt
    intermediate = out if intermediate_fmt == fmt else out.with_suffix(
        out.suffix + f".{intermediate_fmt}.part"
    )

    if intermediate_fmt == "srt":
        codec_args = ["-c:s", "srt"]
    elif intermediate_fmt in {"ass", "ssa"}:
        codec_args = ["-c:s", "ass"]
    elif intermediate_fmt == "vtt":
        codec_args = ["-c:s", "webvtt"]
    else:
        codec_args = ["-c", "copy"]

    cmd = [
        ffmpeg, "-y", "-i", str(path),
        "-map", selector, *codec_args, str(intermediate),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        fallback = out.with_suffix(out.suffix + ".copy.part")
        cmd2 = [
            ffmpeg, "-y", "-i", str(path),
            "-map", selector, "-c", "copy", str(fallback),
        ]
        proc2 = _run(cmd2)
        if proc2.returncode != 0:
            raise ToolError((proc.stderr or proc2.stderr or "ffmpeg failed")[-400:])
        intermediate = fallback

    if intermediate.resolve() != out.resolve():
        raw = intermediate.read_text(encoding="utf-8", errors="replace")
        try:
            src_fmt = detect_format(intermediate, raw)
        except ValueError:
            src_fmt = intermediate_fmt if intermediate_fmt in FORMATS else "srt"
        if fmt in FORMATS:
            converted = convert_text(raw, fmt, source=src_fmt)
            out.write_bytes(converted.text.encode("utf-8"))
        else:
            intermediate.replace(out)
        try:
            intermediate.unlink(missing_ok=True)
        except OSError:
            pass
    return out


def extract_from_video(
    path,
    output_dir=None,
    fmt: str = "srt",
    languages: tuple[str, ...] | None = None,
    indexes: tuple[int, ...] | None = None,
    all_streams: bool = False,
    prefer_text: bool = True,
    dry: bool = False,
    fix=None,
    fix_opts=None,
) -> ExtractResult:
    """Extract subtitle streams from a video file."""
    path = Path(path)
    result = ExtractResult(source=str(path))
    if not path.exists():
        raise FileNotFoundError(path)

    streams = list_subtitle_streams(path)
    result.streams = list(streams)
    if not streams:
        result.message = "no subtitle streams found"
        return result

    fmt = fmt.lower().lstrip(".")
    wanted_langs = {x.lower() for x in languages} if languages is not None else None
    chosen: list[SubtitleStream] = []
    for s in streams:
        if indexes is not None and s.index not in indexes:
            continue
        if wanted_langs is not None:
            lang = (s.language or "").lower()
            if lang not in wanted_langs:
                continue
        if prefer_text and not s.is_text and fmt in FORMATS:
            result.skipped.append(f"{s.label} (image-based)")
            continue
        chosen.append(s)

    if not chosen:
        result.message = "no streams matched filters"
        return result

    if not all_streams and indexes is None:
        text = [s for s in chosen if s.is_text]
        chosen = [text[0] if text else chosen[0]]

    out_dir = Path(output_dir) if output_dir else path.parent
    for s in chosen:
        target_fmt = fmt if s.is_text else CODEC_TO_FORMAT.get(s.codec, "sup")
        try:
            out = extract_stream(
                path, s, streams=streams, fmt=target_fmt,
                out_dir=out_dir, dry=dry,
            )
        except ToolError as e:
            result.skipped.append(f"{s.label}: {e}")
            continue
        result.outputs.append(str(out))
        if dry or fix is None or not s.is_text:
            continue
        p = Path(out)
        fix_path = p
        if p.suffix.lower() != ".srt":
            try:
                conv = convert_file(p, "srt")
                fix_path = Path(conv.path)
                result.converted.append(str(fix_path))
            except Exception as e:  # noqa: BLE001
                result.skipped.append(f"convert {p.name}: {e}")
                continue
        try:
            if fix_opts is None:
                res = fix(fix_path)
            else:
                res = fix(fix_path, fix_opts)
        except TypeError:
            res = fix(fix_path)
        except Exception as e:  # noqa: BLE001
            result.skipped.append(f"fix {fix_path.name}: {e}")
            continue
        if res is not None:
            result.fixed.append(str(fix_path))

    if not result.message:
        result.message = f"extracted {len(result.outputs)} stream(s)"
    return result


def collect_videos(paths, recursive: bool = True) -> list[Path]:
    """Expand files/directories into a list of video paths."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            for child in sorted(iterator):
                if child.is_file() and is_video(child):
                    found.append(child)
        elif p.is_file():
            found.append(p)
    return found
