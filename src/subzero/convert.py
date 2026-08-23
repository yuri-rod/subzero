"""Convert between common subtitle formats.

Supported formats: SRT, WebVTT (VTT), ASS/SSA. Conversion is pure Python so it
works without ffmpeg; extraction from video containers still needs ffmpeg.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .core import CUE, TIMECODE, normalise

FORMATS = ("srt", "vtt", "ass", "ssa")
EXT_TO_FORMAT = {
    ".srt": "srt",
    ".vtt": "vtt",
    ".webvtt": "vtt",
    ".ass": "ass",
    ".ssa": "ssa",
}

_VTT_CUE = re.compile(
    rf"((?:{TIMECODE})\s*-->\s*(?:{TIMECODE})[^\n]*)\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.S,
)
_ASS_DIALOGUE = re.compile(
    r"^(Dialogue|Comment)\s*:\s*(?:marked=)?\d*,"
    r"([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),(.*)$",
    re.M | re.I,
)
_TS = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:([.,])(\d{1,3}))?$"
)


@dataclass(frozen=True)
class Cue:
    """One timed subtitle block in a format-neutral shape."""

    start: str  # HH:MM:SS,mmm (SRT-style comma)
    end: str
    text: str
    style: str = "Default"


@dataclass
class ConvertResult:
    text: str
    cues: int
    source_format: str
    target_format: str
    path: str | None = None


def detect_format(path_or_text: str | Path, text: str | None = None) -> str:
    """Guess format from extension first, then from content."""
    p = Path(str(path_or_text))
    ext = p.suffix.lower()
    if ext in EXT_TO_FORMAT:
        return EXT_TO_FORMAT[ext]
    sample = (text if text is not None else "")[:2000].lstrip("\ufeff")
    head = sample.lstrip().lower()
    if head.startswith("webvtt"):
        return "vtt"
    if "[script info]" in head or "dialogue:" in head:
        return "ass"
    if "-->" in sample:
        return "srt"
    raise ValueError(f"cannot detect subtitle format for {path_or_text}")


def to_srt_time(value: str) -> str:
    """Normalise any common timestamp to SRT ``HH:MM:SS,mmm``."""
    value = value.strip()
    m = _TS.match(value)
    if not m:
        raise ValueError(f"bad timestamp: {value!r}")
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    sep, frac = m.group(4), m.group(5)
    if not frac:
        ms = "000"
    elif sep == "." and len(frac) == 2:
        # ASS centiseconds
        ms = f"{int(frac):02d}0"
    else:
        ms = (frac + "000")[:3]
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms}"


def to_vtt_time(value: str) -> str:
    return to_srt_time(value).replace(",", ".")


def to_ass_time(value: str) -> str:
    """ASS uses ``H:MM:SS.cs`` (centiseconds, single-digit hours ok)."""
    srt = to_srt_time(value)
    h, m, rest = srt.split(":")
    sec, ms = rest.split(",")
    cs = min(99, int(round(int(ms) / 10)))
    return f"{int(h)}:{m}:{sec}.{cs:02d}"


def _clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_srt(raw: str) -> list[Cue]:
    raw = normalise(raw)
    cues: list[Cue] = []
    for timing, body in CUE.findall(raw):
        left, _, right = timing.partition("-->")
        start = left.strip().split()[0] if left.strip() else left.strip()
        end = right.strip().split()[0] if right.strip() else right.strip()
        text = _clean_text(body)
        if not text:
            continue
        cues.append(Cue(start=to_srt_time(start), end=to_srt_time(end), text=text))
    return cues


def parse_vtt(raw: str) -> list[Cue]:
    raw = normalise(raw)
    body = raw
    if body.lstrip().lower().startswith("webvtt"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
    body = re.sub(
        r"(?im)^(NOTE|STYLE|REGION).*?(?=\n\s*\n|\Z)", "", body, flags=re.S
    )
    cues: list[Cue] = []
    for timing, text in _VTT_CUE.findall(body):
        left, _, right = timing.partition("-->")
        start = left.strip().split()[0]
        end = right.strip().split()[0]
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        cues.append(Cue(start=to_srt_time(start), end=to_srt_time(end), text=cleaned))
    return cues


def _ass_override_to_plain(text: str) -> str:
    text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    text = re.sub(r"\{[^}]*\}", "", text)
    return _clean_text(text)


def parse_ass(raw: str) -> list[Cue]:
    raw = normalise(raw)
    cues: list[Cue] = []
    for m in _ASS_DIALOGUE.finditer(raw):
        kind = m.group(1).lower()
        if kind == "comment":
            continue
        start, end = m.group(2).strip(), m.group(3).strip()
        style = m.group(4).strip() or "Default"
        text = _ass_override_to_plain(m.group(10))
        if not text:
            continue
        cues.append(
            Cue(
                start=to_srt_time(start),
                end=to_srt_time(end),
                text=text,
                style=style,
            )
        )
    return cues


def parse_subtitles(raw: str, fmt: str | None = None) -> list[Cue]:
    fmt = (fmt or detect_format("unknown", raw)).lower()
    if fmt == "srt":
        return parse_srt(raw)
    if fmt == "vtt":
        return parse_vtt(raw)
    if fmt in {"ass", "ssa"}:
        return parse_ass(raw)
    raise ValueError(f"unsupported subtitle format: {fmt}")


def render_srt(cues: list[Cue]) -> str:
    parts = []
    for i, c in enumerate(cues, 1):
        parts.append(f"{i}\n{c.start} --> {c.end}\n{c.text}\n")
    return "\r\n".join(parts) + ("\r\n" if parts else "")


def render_vtt(cues: list[Cue]) -> str:
    lines = ["WEBVTT", ""]
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{to_vtt_time(c.start)} --> {to_vtt_time(c.end)}")
        lines.append(c.text.replace("\r\n", "\n"))
        lines.append("")
    return "\n".join(lines)


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def render_ass(cues: list[Cue]) -> str:
    lines = [_ASS_HEADER.rstrip()]
    for c in cues:
        text = c.text.replace("\r\n", "\n").replace("\n", "\\N")
        style = c.style or "Default"
        lines.append(
            f"Dialogue: 0,{to_ass_time(c.start)},{to_ass_time(c.end)},"
            f"{style},,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def render(cues: list[Cue], fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "srt":
        return render_srt(cues)
    if fmt == "vtt":
        return render_vtt(cues)
    if fmt in {"ass", "ssa"}:
        return render_ass(cues)
    raise ValueError(f"unsupported subtitle format: {fmt}")


def convert_text(
    raw: str,
    target: str,
    source: str | None = None,
) -> ConvertResult:
    """Convert subtitle text from one format to another."""
    source = (source or detect_format("in-memory", raw)).lower()
    target = target.lower().lstrip(".")
    if target not in FORMATS:
        raise ValueError(f"unsupported target format: {target}")
    cues = parse_subtitles(raw, source)
    if not cues:
        raise ValueError("no subtitle cues found")
    return ConvertResult(
        text=render(cues, target),
        cues=len(cues),
        source_format=source,
        target_format=target,
    )


def convert_file(
    path,
    target: str,
    output=None,
    source: str | None = None,
    source_fmt: str | None = None,
    dry: bool = False,
) -> ConvertResult:
    """Convert a subtitle file. Writes next to the source unless *output* is set."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("subzero", raw_bytes, 0, 1, "no supported encoding")
    src = (source or source_fmt or detect_format(path, text)).lower()
    target = target.lower().lstrip(".")
    result = convert_text(text, target, source=src)
    if output is not None:
        out = Path(output)
    else:
        ext = "ssa" if target == "ssa" else target
        out = path.with_suffix(f".{ext}")
    result.path = str(out)
    if not dry:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(result.text.encode("utf-8"))
    return result

dump_srt = render_srt
dump_vtt = render_vtt
dump_ass = render_ass
