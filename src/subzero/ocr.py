"""Speech gap analysis and burned-in open caption OCR recovery."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .convert import Cue, dump_srt, parse_srt
from .core import Options, fix_text, read
from .reference import build_reference
from .shift import TIME, _to_seconds
from .timing import spans
from .translate import OllamaClient, OpenAIClient, translate_cues


@dataclass
class GapReport:
    total_gaps: int
    speech_seconds: float
    cues_recovered: int
    cues: list[Cue]
    target: str | None = None


DEFAULT_IGNORE_PATTERNS = (
    re.compile(r"^(?:(?:tuku|lavo|gata|vatu|civa)(?:\s+tribe)?|medical|survivor)$", re.I),
    re.compile(r"^(?:[A-Z]+\s+)?DAY\s+\d+$", re.I),
    re.compile(r"^(?:(?:(?:co-)?executive|associate|supervising)\s+)?(?:producers?|directors?)$", re.I),
    re.compile(r"^(?:hosted|created|produced|directed|written)\s+by$", re.I),
    re.compile(r"^(?:global|international)\s+content\s+distribution$", re.I),
)


def get_vision_ocr_bin() -> str | None:
    env_bin = os.getenv("SUBZERO_VISION_OCR")
    if env_bin and Path(env_bin).is_file() and os.access(env_bin, os.X_OK):
        return env_bin

    which_bin = shutil.which("vision_ocr")
    if which_bin:
        return which_bin

    swift_src = Path(__file__).resolve().parent / "vision_ocr.swift"
    if not swift_src.is_file():
        swift_src = Path(__file__).resolve().parent.parent.parent / "tools" / "vision_ocr.swift"

    repo_bin = Path(__file__).resolve().parent.parent.parent / "tools" / "vision_ocr"
    if (repo_bin.is_file() and os.access(repo_bin, os.X_OK)
            and (not swift_src.is_file() or repo_bin.stat().st_mtime >= swift_src.stat().st_mtime)):
        return str(repo_bin)

    if swift_src.is_file() and sys.platform == "darwin":
        cache_bin = Path.home() / ".cache" / "subzero" / "bin" / "vision_ocr"
        if cache_bin.is_file() and os.access(cache_bin, os.X_OK):
            if cache_bin.stat().st_mtime >= swift_src.stat().st_mtime:
                return str(cache_bin)
        if shutil.which("swiftc"):
            cache_bin.parent.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(
                ["swiftc", "-O", str(swift_src), "-o", str(cache_bin)],
                capture_output=True,
                check=False,
            )
            if res.returncode == 0 and cache_bin.is_file():
                cache_bin.chmod(0o755)
                return str(cache_bin)
    return None


def find_speech_gaps(
    video: str | Path,
    subtitle_content: str,
    min_duration: float = 1.0,
    max_coverage: float = 0.15,
    cache_dir: str | Path | None = None,
) -> list[tuple[float, float]]:
    cache = cache_dir or Path.home() / ".cache" / "subzero" / "references"
    ref = build_reference(video, cache)
    return uncovered_intervals(ref.get("speech", []), spans(subtitle_content), min_duration)


def uncovered_intervals(
    intervals: Iterable[tuple[float, float]],
    covered: Iterable[tuple[float, float]],
    min_duration: float = 0.5,
) -> list[tuple[float, float]]:
    subtitles = sorted(covered)
    gaps = []
    for start, end in sorted(intervals):
        cursor = start
        for sub_start, sub_end in subtitles:
            if sub_end <= cursor:
                continue
            if sub_start >= end:
                break
            if sub_start - cursor >= min_duration:
                gaps.append((round(cursor, 3), round(sub_start, 3)))
            cursor = max(cursor, sub_end)
            if cursor >= end:
                break
        if end - cursor >= min_duration:
            gaps.append((round(cursor, 3), round(end, 3)))
    return gaps


def find_caption_gaps(
    video: str | Path,
    subtitle_content: str,
    min_duration: float = 0.5,
    cache_dir: str | Path | None = None,
) -> list[tuple[float, float]]:
    cache = cache_dir or Path.home() / ".cache" / "subzero" / "references"
    ref = build_reference(video, cache)
    duration = float(ref.get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Video duration is missing from the caption reference")
    occupied = []
    for cue in parse_srt(subtitle_content):
        stamp = TIME.search(f"{cue.start} --> {cue.end}")
        if stamp is not None:
            occupied.append((_to_seconds(*stamp.groups()[:4]), _to_seconds(*stamp.groups()[4:])))
    return uncovered_intervals([(0.0, duration)], occupied, min_duration)


def clean_ocr_text(
    text: str,
    ignore_patterns: Iterable[re.Pattern] = DEFAULT_IGNORE_PATTERNS,
) -> str:
    text = re.sub(r"[ạẠỊ]", "", text)
    text = re.sub(r"\bl(?=['’]ve\b)", "I", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t\r\n'\"`´‘“”)({}[]").strip()
        if not line or not any(char.isalpha() for char in line):
            continue
        if any(pat.search(line) for pat in ignore_patterns):
            continue
        lines.append(line)
    return "\n".join(lines)


def caption_text(frame: dict, center_tolerance: float | None = 0.12) -> str:
    if not frame.get("items"):
        return frame["subtitleText"]
    lines = []
    for region in frame["items"]:
        if not isinstance(region, dict) or not isinstance(region.get("text"), str):
            raise RuntimeError("Vision OCR returned an invalid text region")
        try:
            confidence, x, y, width, height = [float(region[name]) for name in
                                              ("confidence", "x", "y", "width", "height")]
        except (KeyError, TypeError, ValueError) as err:
            raise RuntimeError("Vision OCR returned invalid text coordinates") from err
        if not all(math.isfinite(value) for value in (confidence, x, y, width, height)):
            raise RuntimeError("Vision OCR returned invalid text coordinates")
        if confidence >= 0.8 and height >= 0.12 and width >= 0.45 and y >= 0.25 and abs(x + width / 2 - 0.5) <= 0.12:
            return ""
        if (confidence >= 0.8 and 0.03 <= y <= 0.25 and 0.045 <= height <= 0.10
                and width >= 0.04 and (center_tolerance is None
                                      or abs(x + width / 2 - 0.5) <= center_tolerance)):
            lines.append((y, x, region["text"]))
    lines.sort(key=lambda region: (-region[0], region[1]))
    return "\n".join(text for _, _, text in lines)


def fmt_srt_time(sec: float) -> str:
    millis = max(0, round(sec * 1000))
    seconds, millis = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _caption_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.lower().replace("’", "'"))


def _single_character_change(first: str, second: str) -> bool:
    if len(first) == len(second):
        return sum(left != right for left, right in zip(first, second)) == 1
    short, long = sorted((first, second), key=len)
    if len(long) - len(short) != 1:
        return False
    for index, (left, right) in enumerate(zip(short, long)):
        if left != right:
            return short[index:] == long[index + 1:]
    return True


def cluster_ocr_detections(
    detections: list[tuple[float, str]],
    min_duration: float = 1.4,
    max_gap: float = 2.0,
    fuzzy_ratio: float = 0.92,
    sample_duration: float = 1.0,
) -> list[Cue]:
    ordered = sorted(detections)
    stable = list(ordered)
    frame_gap = min(max_gap, sample_duration * 1.5)
    negations = {"no", "not", "never", "neither", "nor", "nobody", "nothing", "nowhere", "none", "without", "cannot"}
    for index in range(1, len(ordered) - 1):
        before, middle, after = ordered[index - 1:index + 2]
        if not (0 < middle[0] - before[0] <= frame_gap and 0 < after[0] - middle[0] <= frame_gap):
            continue
        first_words, middle_words, last_words = [_caption_words(text) for _, text in (before, middle, after)]
        if min(len(first_words), len(middle_words)) < 5 or first_words != last_words:
            continue
        first_negative = [word for word in first_words if word in negations or word.endswith("n't")]
        middle_negative = [word for word in middle_words if word in negations or word.endswith("n't")]
        if first_negative == middle_negative and _single_character_change(" ".join(first_words), " ".join(middle_words)):
            stable[index] = (middle[0], before[1])
    merged: list[tuple[float, float, str]] = []
    current = None
    for timestamp, text in stable:
        text = text.strip()
        if current is not None:
            start, end, previous, last_seen = current
            left, right = " ".join(previous.lower().split()), " ".join(text.lower().split())
            left_words, right_words = _caption_words(left), _caption_words(right)
            short, long = sorted((left, right), key=len)
            fragment = (len(short.split()) >= 3 and not short.endswith((".", "?", "!"))
                        and short in long and len(short) >= len(long) * 0.55)
            if text and timestamp - last_seen <= max_gap and (left_words == right_words or fragment):
                current = (start, timestamp + sample_duration,
                           text if len(text) > len(previous) else previous, timestamp)
                continue
            merged.append((start, min(max(end, start + min_duration), timestamp), previous))
        current = (timestamp, timestamp + sample_duration, text, timestamp) if text else None
    if current is not None:
        start, end, text, _ = current
        merged.append((start, max(end, start + min_duration), text))
    return [Cue(fmt_srt_time(start), fmt_srt_time(end), text)
            for start, end, text in merged if end > start]


def extract_and_ocr_gaps(
    video: str | Path,
    gaps: list[tuple[float, float]],
    ocr_bin: str | None = None,
    fps: float = 2.0,
    tmp_dir: Path | None = None,
    ignore_patterns: Iterable[re.Pattern] = DEFAULT_IGNORE_PATTERNS,
    progress: Callable[[int, int], None] | None = None,
    center_tolerance: float | None = 0.12,
) -> list[Cue]:
    if not gaps:
        return []

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("OCR frame rate must be positive")
    binary = ocr_bin or get_vision_ocr_bin()
    if not binary:
        raise RuntimeError(
            "vision_ocr tool not found. Compile tools/vision_ocr.swift or set SUBZERO_VISION_OCR."
        )

    recovered = []
    cleanup_tmp = tmp_dir is None
    work_dir = tmp_dir or Path(tempfile.mkdtemp(prefix="subzero_ocr_"))

    try:
        for idx, (g_start, g_end) in enumerate(gaps, start=1):
            if g_end <= g_start:
                continue
            gap_dir = Path(tempfile.mkdtemp(prefix=f"gap_{idx:03d}_", dir=work_dir))
            dur = g_end - g_start
            cmd = [
                "ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", f"{g_start:.3f}", "-i", str(video),
                "-t", f"{dur:.3f}", "-an", "-sn", "-vf", f"fps={fps}:start_time=0,showinfo", "-q:v", "2",
                str(gap_dir / "f_%03d.jpg"),
            ]
            try:
                decoded = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                         text=True, check=False, timeout=180)
                if decoded.returncode:
                    raise RuntimeError(f"ffmpeg failed during caption extraction: {decoded.stderr[-400:]}")
                frames = sorted(gap_dir.glob("*.jpg"), key=lambda frame: int(frame.stem[2:]))
                timestamps = [float(value) for value in re.findall(r"\bpts_time:([\d.eE+-]+)", decoded.stderr)]
                if not frames or len(timestamps) < len(frames):
                    raise RuntimeError("ffmpeg did not return timestamped caption frames")
                res = subprocess.run([binary, "--json"] + [str(frame) for frame in frames],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     check=False, timeout=max(120, len(frames) * 10))
                if res.returncode:
                    raise RuntimeError(f"Vision OCR failed: {res.stderr[-400:]}")
                try:
                    rows = json.loads(res.stdout)
                except (ValueError, UnicodeError) as err:
                    raise RuntimeError("Vision OCR returned invalid JSON") from err
                if not isinstance(rows, list) or len(rows) != len(frames):
                    raise RuntimeError("Vision OCR did not return every caption frame")
                by_file = {}
                for row in rows:
                    if (not isinstance(row, dict) or not isinstance(row.get("file"), str)
                            or not isinstance(row.get("subtitleText"), str) or row.get("error")):
                        raise RuntimeError("Vision OCR returned an invalid frame")
                    name = Path(row["file"]).name
                    if name in by_file:
                        raise RuntimeError("Vision OCR returned a duplicate frame")
                    by_file[name] = caption_text(row, center_tolerance)
                if set(by_file) != {frame.name for frame in frames}:
                    raise RuntimeError("Vision OCR returned unexpected caption frames")
                detections = [(g_start + timestamp, clean_ocr_text(by_file[frame.name], ignore_patterns))
                              for frame, timestamp in zip(frames, timestamps) if timestamp < dur]
                cues = cluster_ocr_detections(detections, min_duration=1 / fps,
                                             max_gap=1.5 / fps, sample_duration=1 / fps)
                recovered.extend(clip_cues(cues, [(g_start, g_end)]))
            except subprocess.TimeoutExpired as err:
                raise RuntimeError(f"{err.cmd[0]} timed out during caption extraction") from err
            finally:
                shutil.rmtree(gap_dir)
            if progress:
                progress(idx, len(gaps))
    finally:
        if cleanup_tmp and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    return recovered


def cue_start_seconds(c: Cue) -> float:
    m = TIME.search(f"{c.start} --> {c.end}")
    if not m:
        return 0.0
    return _to_seconds(*m.groups()[:4])


def clip_cues(cues: list[Cue], intervals: list[tuple[float, float]]) -> list[Cue]:
    clipped = []
    seen = set()
    last_end = 0.0
    for cue in sorted(cues, key=cue_start_seconds):
        stamp = TIME.search(f"{cue.start} --> {cue.end}")
        if stamp is None:
            raise ValueError("Recovered caption has invalid timestamps")
        start, end = _to_seconds(*stamp.groups()[:4]), _to_seconds(*stamp.groups()[4:])
        candidates = [(max(start, left, last_end), min(end, right)) for left, right in intervals]
        left, right = max(candidates, key=lambda bounds: bounds[1] - bounds[0], default=(0, 0))
        signature = (round(left, 3), round(right, 3), cue.text)
        if right - left < 0.1 or signature in seen:
            continue
        clipped.append(Cue(fmt_srt_time(left), fmt_srt_time(right), cue.text, cue.style))
        seen.add(signature)
        last_end = right
    return clipped


def fill_subtitle_gaps(
    video: str | Path,
    subtitle_path: str | Path,
    output: str | Path | None = None,
    target_lang: str | None = None,
    provider: str = "ollama",
    model: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    dry_run: bool = False,
    backup: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> GapReport:
    sub_p = Path(subtitle_path)
    if sub_p.is_symlink() or not sub_p.is_file():
        raise RuntimeError("Caption recovery requires a regular subtitle file")
    before = sub_p.stat()
    original = sub_p.read_bytes()
    content, _ = read(sub_p)
    existing_cues = parse_srt(content)
    if not existing_cues:
        raise ValueError("Subtitle file has no valid cues")
    gaps = find_caption_gaps(video, content, cache_dir=cache_dir)
    total_speech_sec = sum(end - start for start, end in gaps)

    if not gaps:
        return GapReport(total_gaps=0, speech_seconds=0.0, cues_recovered=0, cues=[])

    recovered = extract_and_ocr_gaps(video, gaps, progress=progress)
    if not recovered:
        return GapReport(total_gaps=len(gaps), speech_seconds=total_speech_sec, cues_recovered=0, cues=[])

    to_merge = recovered
    if target_lang and target_lang.lower() not in ("en", "eng"):
        if provider.lower() in ("openai", "openrouter", "groq", "deepseek"):
            base_url = url or (
                "https://openrouter.ai/api/v1" if provider.lower() == "openrouter" else
                "https://api.groq.com/openai/v1" if provider.lower() == "groq" else
                "https://api.deepseek.com/v1" if provider.lower() == "deepseek" else
                os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            chosen_model = model or "gpt-4o-mini"
            client = OpenAIClient(api_key=api_key, base_url=base_url, model=chosen_model)
        else:
            chosen_url = url or "http://127.0.0.1:11434"
            chosen_model = model or "subzero/hy-mt2:7b"
            client = OllamaClient(url=chosen_url, model=chosen_model)

        to_merge = translate_cues(
            recovered,
            target_lang=target_lang,
            client=client,
            source_lang="en",
            progress=progress,
        )
        if len(to_merge) != len(recovered):
            raise RuntimeError("Caption translation did not preserve every recovered cue")
        english = {"the", "you", "your", "that", "with", "this", "they", "have", "what",
                   "were", "about", "there", "would", "don't", "can't", "yes", "no"}
        for source, translated in zip(recovered, to_merge):
            src_words = re.findall(r"[a-zA-Z']+", source.text.lower())
            dst_words = re.findall(r"[a-zA-Z']+", translated.text.lower())
            if (target_lang.lower().startswith("pt") and src_words == dst_words
                    and (len(src_words) >= 2 or english.intersection(src_words))):
                raise RuntimeError(f"Caption translation returned untranslated English: {source.text[:100]}")
        to_merge = [Cue(source.start, source.end, translated.text, source.style)
                    for source, translated in zip(recovered, to_merge)]

    to_merge = clip_cues(to_merge, gaps)
    to_merge = parse_srt(fix_text(dump_srt(to_merge), Options(max_line=42, preserve_breaks=False)).text)

    out_path = Path(output) if output else sub_p
    if not dry_run and to_merge:
        if out_path.is_symlink():
            raise RuntimeError("Refusing to replace a subtitle symlink")
        rendered = dump_srt(sorted(existing_cues + to_merge, key=cue_start_seconds))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out_path.parent,
                                         prefix=".subtitle-", suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            try:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            except (OSError, UnicodeError):
                tmp.unlink(missing_ok=True)
                raise
        try:
            current = sub_p.lstat()
            if (not stat.S_ISREG(current.st_mode) or not os.path.samestat(before, current)
                    or current.st_mtime_ns != before.st_mtime_ns or sub_p.read_bytes() != original):
                raise RuntimeError("Subtitle changed during caption recovery")
            if out_path.is_symlink():
                raise RuntimeError("Refusing to replace a subtitle symlink")
            if backup and out_path.resolve() == sub_p.resolve():
                bak = sub_p.with_suffix(".srt.bak")
                try:
                    with bak.open("xb") as handle:
                        handle.write(original)
                except FileExistsError:
                    pass
            tmp.chmod(stat.S_IMODE(before.st_mode))
            os.replace(tmp, out_path)
        finally:
            tmp.unlink(missing_ok=True)

    return GapReport(
        total_gaps=len(gaps),
        speech_seconds=total_speech_sec,
        cues_recovered=len(to_merge),
        cues=to_merge,
        target=str(out_path),
    )
