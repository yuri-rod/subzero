"""Universal subtitle & audio AI toolkit."""

__version__ = "1.0.0"

from .convert import (
    Cue,
    ConvertResult,
    convert_file,
    convert_text,
    detect_format,
    dump_ass,
    dump_srt,
    dump_vtt,
    parse_ass,
    parse_srt,
    parse_vtt,
)
from .core import Options, Result, Stats, analyze, fix_file, fix_text, keep_breaks, rewrap, strip_sdh
from .extract import (
    ExtractResult,
    SubtitleStream,
    ToolError,
    collect_videos,
    extract_from_video,
    list_subtitle_streams,
)
from .moviehash import moviehash
from .shift import shift_file, shift_timestamps
from .sync import auto_sync_file, probe_audio_delay
from .translate import OllamaClient, translate_cues, translate_file

__all__ = [
    "__version__",
    "Cue",
    "ExtractResult",
    "ConvertResult",
    "OllamaClient",
    "Options",
    "Result",
    "Stats",
    "SubtitleStream",
    "ToolError",
    "analyze",
    "auto_sync_file",
    "collect_videos",
    "convert_file",
    "convert_text",
    "detect_format",
    "dump_ass",
    "dump_srt",
    "dump_vtt",
    "extract_from_video",
    "fix_file",
    "fix_text",
    "keep_breaks",
    "rewrap",
    "strip_sdh",
    "list_subtitle_streams",
    "moviehash",
    "parse_ass",
    "parse_srt",
    "parse_vtt",
    "probe_audio_delay",
    "shift_file",
    "shift_timestamps",
    "translate_cues",
    "translate_file",
]
