"""Speech gap analysis and burned-in open caption OCR recovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

from .caption_quality import validate_caption_readings
from .caption_rescue import CAPTION_CROP, CaptionFrame, CaptionRescue, _image_bytes, crop_caption_image
from .caption_scan_cache import CaptionScanCache
from .compute import compute_lease_fd, compute_phase
from .convert import Cue, dump_srt, parse_srt
from .core import Options, fix_text, read
from .reference import build_reference, fingerprint
from .shift import TIME, _to_seconds
from .timing import spans
from .translate import OllamaClient, OpenAIClient, translate_cues


CAPTION_SCAN_VERSION = 8


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
    elif sys.platform == "darwin":
        cache_bin = Path.home() / ".cache" / "subzero" / "bin" / "vision_ocr"
        if cache_bin.is_file() and os.access(cache_bin, os.X_OK):
            return str(cache_bin)
    return None


def _caption_scan_cache(video: str | Path, root: Path) -> CaptionScanCache:
    binary = get_vision_ocr_bin()
    if not binary:
        raise RuntimeError("Apple Vision OCR is unavailable for caption scanning")
    try:
        probe = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                               check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise RuntimeError("Cannot identify FFmpeg for the caption scan cache") from err
    if probe.returncode or not probe.stdout.strip():
        raise RuntimeError("FFmpeg returned no version for the caption scan cache")
    recognition = json.dumps({
        "version": CAPTION_SCAN_VERSION,
        "vision": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "macos": platform.mac_ver()[0],
        "ffmpeg": probe.stdout.splitlines()[0],
    }, sort_keys=True)
    return CaptionScanCache(root, fingerprint(video), recognition)


def find_speech_gaps(
    video: str | Path,
    subtitle_content: str,
    min_duration: float = 1.0,
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
    ignore_patterns = tuple(ignore_patterns)
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
    cleaned = "\n".join(lines)
    if any(pattern.search(" ".join(cleaned.split())) for pattern in ignore_patterns):
        return ""
    return cleaned


def _caption_ink(region: dict) -> float:
    ink = region.get("captionInk", 1)
    if isinstance(ink, bool) or not isinstance(ink, (int, float)):
        raise RuntimeError("Vision OCR returned invalid caption ink")
    if not math.isfinite(ink) or not 0 <= ink <= 1:
        raise RuntimeError("Vision OCR returned invalid caption ink")
    return ink


def _caption_geometry(region: dict) -> dict:
    if not isinstance(region, dict) or not isinstance(region.get("text"), str):
        raise RuntimeError("Vision OCR returned an invalid text region")
    try:
        coordinates = {name: float(region[name]) for name in ("confidence", "x", "y", "width", "height")}
        coordinates["angle"] = float(region.get("angle", 0))
    except (KeyError, TypeError, ValueError) as err:
        raise RuntimeError("Vision OCR returned invalid text coordinates") from err
    if not all(math.isfinite(value) for value in coordinates.values()):
        raise RuntimeError("Vision OCR returned invalid text coordinates")
    if "angle" not in region:
        coordinates.pop("angle")
    return {**region, **coordinates}


def _caption_regions(frame: dict, center_tolerance: float | None, max_height: float = 0.10) -> list[dict]:
    if _caption_credit_layout(frame) or _caption_logo_layout(frame):
        return []
    regions = []
    for region in frame.get("items", []):
        region = _caption_geometry(region)
        confidence, x, y, width, height, angle = [region.get(name, 0) for name in
                                                ("confidence", "x", "y", "width", "height", "angle")]
        if abs(angle) > 10 or _caption_ink(region) < 0.04:
            continue
        if (confidence >= 0.8 and height >= 0.12 and abs(x + width / 2 - 0.5) <= 0.12
                and ((width >= 0.45 and y >= 0.25) or (width >= 0.35 and y >= 0.45))):
            return []
        if (confidence >= 0.8 and 0.03 <= y <= 0.19 and 0.045 <= height <= max_height
                and width >= 0.04 and (center_tolerance is None
                                      or abs(x + width / 2 - 0.5) <= center_tolerance)):
            regions.append({**region, "x": x, "y": y, "width": width, "height": height})
    return regions


def _same_caption_position(first: dict, second: dict) -> bool:
    return (abs(first["x"] + first["width"] / 2 - second["x"] - second["width"] / 2) <= 0.03
            and abs(first["y"] + first["height"] / 2 - second["y"] - second["height"] / 2) <= 0.04
            and 0.75 <= first["width"] / second["width"] <= 1.25)


def _caption_credit_layout(frame: dict) -> bool:
    rows = []
    for region in frame.get("items", []):
        region = _caption_geometry(region)
        if "captionInk" not in region or _caption_ink(region) < 0.04:
            continue
        if (float(region["confidence"]) >= 0.8 and abs(float(region.get("angle", 0))) <= 10
                and 0.03 <= float(region["y"]) <= 0.35 and 0.025 <= float(region["height"]) <= 0.12):
            center = float(region["y"]) + float(region["height"]) / 2
            if not any(abs(center - prior) <= 0.025 for prior in rows):
                rows.append(center)
    return len(rows) >= 4


def _caption_title_card(frame: dict) -> bool:
    return _caption_credit_layout(frame) or _caption_logo_layout(frame) or any(float(region["confidence"]) >= 0.8 and abs(float(region.get("angle", 0))) <= 10
               and float(region["height"]) >= 0.12
               and abs(float(region["x"]) + float(region["width"]) / 2 - 0.5) <= 0.12
               and ((float(region["width"]) >= 0.45 and float(region["y"]) >= 0.25)
                    or (float(region["width"]) >= 0.35 and float(region["y"]) >= 0.45))
               for region in frame.get("items", []))


def _caption_logo_layout(frame: dict) -> bool:
    regions = [_caption_geometry(region) for region in frame.get("items", [])]
    for title in regions:
        center = title["x"] + title["width"] / 2
        if not (title["confidence"] >= 0.8 and abs(title.get("angle", 0)) <= 10
                and title["height"] >= 0.18 and title["width"] >= 0.35
                and 0.25 <= title["y"] <= 0.65 and abs(center - 0.5) <= 0.12):
            continue
        rows = []
        # Cropping first can leave only the lowest word of a stacked logo.
        for region in regions:
            if (region["confidence"] >= 0.8 and abs(region.get("angle", 0)) <= 10
                    and region["text"].isupper() and 0.025 <= region["height"] <= 0.10
                    and 0.03 <= region["y"] <= 0.30 and region["width"] >= 0.04
                    and region["y"] + region["height"] <= title["y"] + 0.01
                    and abs(region["x"] + region["width"] / 2 - center) <= 0.06):
                row_center = region["y"] + region["height"] / 2
                if not any(abs(row_center - prior) <= 0.025 for prior in rows):
                    rows.append(row_center)
        if len(rows) >= 2:
            return True
    return False


def caption_retry_indices(frames: list[dict], timestamps: list[float], fps: float = 2) -> list[int]:
    texts = [clean_ocr_text(caption_text(frame,
             previous=frames[index - 1] if index else None,
             following=frames[index + 1] if index + 1 < len(frames) else None))
             if frame.get("items") else "" for index, frame in enumerate(frames)]
    selected = set()
    for index in range(len(frames) - 1):
        if not 0 < timestamps[index + 1] - timestamps[index] <= 1.5 / fps:
            continue
        first, second = [_caption_words(text) for text in texts[index:index + 2]]
        if first == second:
            continue
        if first and second and SequenceMatcher(None, " ".join(first), " ".join(second)).ratio() >= 0.65:
            selected.update(range(max(0, index - 1), min(len(frames), index + 3)))
        elif index and not first and second == _caption_words(texts[index - 1]):
            selected.update((index - 1, index, index + 1))
    return sorted(index for index in selected if not _caption_title_card(frames[index]))


def _protected_caption_change(first: str, second: str, *, contained_fragment: bool = False) -> bool:
    quoted = first.lstrip().startswith(('"', "'", "“", "‘")) or any(char in first for char in "ạẠỊ")
    first, second = clean_ocr_text(first), clean_ocr_text(second)
    left, right = _caption_words(first), _caption_words(second)
    if ([index for index, word in enumerate(left) if word == "_"]
            != [index for index, word in enumerate(right) if word == "_"]):
        return True
    if _negation_words(left) != _negation_words(right):
        return True
    numbers = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())
    if [word for word in left if word.isdigit() or word in numbers] != [word for word in right if word.isdigit() or word in numbers]:
        return True
    if contained_fragment:
        return False
    if "".join(left).replace("'", "") == "".join(right).replace("'", ""):
        return False
    if _changed_caption_names(first, second):
        contractions = {"it's", "that's", "here's", "there's"}
        if left and right and right[0] in contractions and left[0][1:] == right[0]:
            return _changed_caption_names(first[1:], second)
        if quoted and len(left) == len(right) and all(a == b or a.rstrip("i") == b for a, b in zip(left, right)):
            return False
        return True
    return False


def _trailing_caption_glyph(first: str, second: str) -> bool:
    left, right = _caption_words(first), _caption_words(second)
    for noisy, clean, text in ((left, right, first), (right, left, second)):
        if (_spoken_word_count(clean) >= 3 and len(noisy) == len(clean) + 1 and noisy[:-1] == clean
                and len(noisy[-1]) == 1
                and re.search(r"[.!?…][\"'”’]*[a-z][\"'”’]*$", text)):
            return True
    return False


def _retry_compatible(first: str, second: str) -> bool:
    protected = _protected_caption_change(first, second)
    first, second = clean_ocr_text(first), clean_ocr_text(second)
    left, right = _caption_words(first), _caption_words(second)
    if left == right:
        return True
    if min(_spoken_word_count(left), _spoken_word_count(right)) < 3 or protected:
        return False
    if len(left) != len(right):
        if _trailing_caption_glyph(first, second):
            return True
        return "".join(left).replace("'", "") == "".join(right).replace("'", "")
    changes = 0
    for original, candidate in zip(left, right):
        if original == candidate:
            continue
        short, long = sorted((original, candidate), key=len)
        if not short or not (long.startswith(short) or long.endswith(short)
                             or original.replace("'", "") == candidate.replace("'", "")):
            return False
        changes += max(1, len(long) - len(short))
    return changes <= 3


def _contained_caption_fragment(fragment: dict, line: dict) -> bool:
    words, complete = _caption_words(fragment["text"]), _caption_words(line["text"])
    if not words or len(words) >= len(complete):
        return False
    overlap = min(fragment["x"] + fragment["width"], line["x"] + line["width"]) - max(fragment["x"], line["x"])
    return (overlap >= 0.8 * fragment["width"]
            and abs(fragment["y"] + fragment["height"] / 2 - line["y"] - line["height"] / 2) <= 0.025
            and any(complete[index:index + len(words)] == words
                    for index in range(len(complete) - len(words) + 1)))


def _restore_split_caption_censors(frame: dict, retry: dict, center_tolerance: float | None) -> dict:
    if _caption_title_card(frame):
        return frame
    fragments = _caption_regions(retry, center_tolerance, max_height=0.12)
    replacements = list(frame.get("items", []))
    for line in _caption_regions(frame, center_tolerance, max_height=0.12):
        words = _caption_words(line["text"])
        if "_" in words:
            continue
        row = sorted((fragment for fragment in fragments
                      if abs(fragment["y"] + fragment["height"] / 2 - line["y"] - line["height"] / 2) <= 0.025
                      and fragment["x"] >= line["x"] - 0.03
                      and fragment["x"] + fragment["width"] <= line["x"] + line["width"] + 0.03),
                     key=lambda fragment: fragment["x"])
        if (len(row) < 2 or sum(fragment["width"] for fragment in row) < 0.8 * line["width"]
                or any(left["x"] + left["width"] > right["x"] + 0.005
                       for left, right in zip(row, row[1:]))):
            continue
        text = " ".join(fragment["text"] for fragment in row)
        observed = _caption_words(text)
        if ("_" not in observed or [word for word in observed if word != "_"] != words
                or _protected_caption_change(line["text"], re.sub(r"_+", " ", text))):
            continue
        # The crop split one physical row; appending its suffix duplicates visible words.
        replacements = [{**prior, "text": text, "candidates": []} if prior == line else prior
                        for prior in replacements]
    return {**frame, "items": replacements}


def recover_caption_runs(frames: list[dict], retries: dict[int, dict], timestamps: list[float],
                         center_tolerance: float | None = 0.12) -> list[str]:
    frames = [_restore_split_caption_censors(frame, retries[index], center_tolerance)
              if index in retries else frame for index, frame in enumerate(frames)]
    regions = [_caption_regions(frame, center_tolerance, max_height=0.12) for frame in frames]
    retried = {index: _caption_regions(frame, center_tolerance, max_height=0.12) for index, frame in retries.items()
               if not _caption_title_card(frames[index])}
    frames = list(frames)
    for index, current in enumerate(regions):
        for region in current:
            text = clean_ocr_text(region["text"])
            prefix = re.match(r"^[^\x00-\x7f]+(?=[A-Z][a-z]+)", text)
            if not prefix:
                continue
            stripped = text[prefix.end():]
            for retry in retried.get(index, []):
                if (not _same_caption_position(region, retry)
                        or _caption_words(stripped) != _caption_words(clean_ocr_text(retry["text"]))
                        or _protected_caption_change(stripped, retry["text"])):
                    continue
                support = sum(any(_same_caption_position(region, candidate)
                                  and _caption_words(stripped) == _caption_words(clean_ocr_text(candidate["text"]))
                                  for candidate in retried.get(neighbor, []))
                              for neighbor in range(max(0, index - 6), min(len(frames), index + 7))
                              if abs(timestamps[neighbor] - timestamps[index]) <= 3)
                if support >= 2:
                    frames[index] = {**frames[index], "items": [
                        {**prior, "text": retry["text"], "candidates": []} if prior == region else prior
                        for prior in frames[index]["items"]]}
                    break
        regions[index] = _caption_regions(frames[index], center_tolerance, max_height=0.12)
    updated = []
    for index, frame in enumerate(frames):
        if _caption_title_card(frame):
            updated.append(frame)
            continue
        replacements = list(frame.get("items", []))
        proposals = regions[index] + [region for region in retried.get(index, [])
                                     if not any(_contained_caption_fragment(region, line) for line in regions[index])]
        for region in proposals:
            votes = Counter()
            renderings = Counter()
            cropped = set()
            linked = {}
            for neighbor in range(max(0, index - 6), min(len(frames), index + 7)):
                if abs(timestamps[neighbor] - timestamps[index]) > 3:
                    continue
                seen = set()
                for candidate in regions[neighbor] + retried.get(neighbor, []):
                    text = clean_ocr_text(candidate["text"])
                    if (not text or not _same_caption_position(region, candidate)
                            or not _retry_compatible(region["text"], text)):
                        continue
                    words = tuple(_caption_words(text))
                    renderings[words, text] += 1
                    seen.add(words)
                    if candidate in retried.get(neighbor, []):
                        cropped.add(words)
                votes.update(seen)
                for words in seen:
                    linked.setdefault(words, set()).update(seen)
            reachable = {tuple(_caption_words(clean_ocr_text(region["text"])))}
            pending = list(reachable)
            while pending:
                for words in linked.get(pending.pop(), set()) - reachable:
                    reachable.add(words)
                    pending.append(words)
            eligible = [words for words in cropped & reachable if votes[words] >= 2]
            if not eligible:
                continue
            words = max(eligible, key=lambda words: (votes[words], len(words), -len(" ".join(words)), words))
            chosen = max((text for key, text in renderings if key == words),
                         key=lambda text: (renderings[words, text], -len(text), text))
            overlapping = [prior for prior in _caption_regions({"items": replacements}, center_tolerance, max_height=0.18)
                           if abs(float(prior["y"]) + float(prior["height"]) / 2 - region["y"] - region["height"] / 2) <= 0.04
                           and region["x"] <= float(prior["x"]) + float(prior["width"]) / 2 <= region["x"] + region["width"]]
            if any(_protected_caption_change(prior["text"], chosen,
                   contained_fragment=_contained_caption_fragment(prior, {**region, "text": chosen}))
                   for prior in overlapping):
                continue
            replacements = [prior for prior in replacements if prior not in overlapping]
            replacements.append({**region, "text": chosen, "candidates": []})
        updated.append({**frame, "items": replacements})
    return [clean_ocr_text(caption_text(frame, center_tolerance,
            previous=updated[index - 1] if index else None,
            following=updated[index + 1] if index + 1 < len(updated) else None))
            for index, frame in enumerate(updated)]


def caption_text(frame: dict, center_tolerance: float | None = 0.12, *,
                 previous: dict | None = None, following: dict | None = None) -> str:
    if not frame.get("items"):
        return frame["subtitleText"]
    before = _caption_regions(previous or {}, center_tolerance, max_height=0.12)
    after = _caption_regions(following or {}, center_tolerance, max_height=0.12)
    lines = []
    for region in _caption_regions(frame, center_tolerance, max_height=0.18):
        first_words = {tuple(_caption_words(prior["text"])) for prior in before if _same_caption_position(region, prior)}
        last_words = {tuple(_caption_words(later["text"])) for later in after if _same_caption_position(region, later)}
        text = region["text"]
        words = tuple(_caption_words(text))
        for candidate in region.get("candidates", []):
            if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
                raise RuntimeError("Vision OCR returned an invalid text candidate")
            try:
                confidence = float(candidate["confidence"])
            except (KeyError, TypeError, ValueError) as err:
                raise RuntimeError("Vision OCR returned an invalid candidate confidence") from err
            if not math.isfinite(confidence):
                raise RuntimeError("Vision OCR returned an invalid candidate confidence")
            candidate_words = tuple(_caption_words(candidate["text"]))
            if (text.lstrip().startswith(('"', "'", "“", "‘")) and words and candidate_words
                    and words[0] != candidate_words[0]):
                continue
            changes = sum(max(end_a - start_a, end_b - start_b)
                          for op, start_a, end_a, start_b, end_b in
                          SequenceMatcher(None, " ".join(words), " ".join(candidate_words), autojunk=False).get_opcodes()
                          if op != "equal")
            if (confidence >= 0.8 and min(_spoken_word_count(words), _spoken_word_count(candidate_words)) >= 5
                    and candidate_words in first_words & last_words
                    and changes <= 2 and not _protected_caption_change(text, candidate["text"])):
                text, words = candidate["text"], candidate_words
                break
        if region["height"] > 0.10 and words not in first_words | last_words:
            continue
        lines.append((region["y"] + region["height"] / 2, region["x"], text))
    lines.sort(key=lambda region: (-region[0], region[1]))
    rows = []
    for line in lines:
        if rows and abs(line[0] - rows[-1][0][0]) <= 0.025:
            rows[-1].append(line)
        else:
            rows.append([line])
    return "\n".join(" ".join(text for _, _, text in sorted(row, key=lambda line: line[1])) for row in rows)


def fmt_srt_time(sec: float) -> str:
    millis = max(0, round(sec * 1000))
    seconds, millis = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _caption_words(text: str) -> list[str]:
    # A censorship bar is source content; its rendered width is not a word change.
    return ["_" if token.startswith("_") else token
            for token in re.findall(r"[^\W_]+(?:'[^\W_]+)*|_+", text.lower().replace("’", "'"))]


def _spoken_word_count(words: Iterable[str]) -> int:
    return sum(bool(word.strip("_")) for word in words)


def _negation_words(words: Iterable[str]) -> list[str]:
    negations = {"no", "not", "never", "neither", "nor", "nobody", "nothing", "nowhere", "none", "without", "cannot"}
    return [word for word in words if word in negations or word.endswith("n't")]


def _changed_caption_names(first: str, second: str) -> bool:
    left, right = [re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.replace("’", "'")) for text in (first, second)]
    first_person = {"I", "I'm", "I've", "I'll", "I'd"}
    left_names = {word.lower() for word in left[1:] if word[0].isupper() and word not in first_person}
    right_names = {word.lower() for word in right[1:] if word[0].isupper() and word not in first_person}
    if left_names != right_names:
        return True
    if left and right and left[0][0].isupper() and right[0][0].isupper() and left[0].lower() != right[0].lower():
        return left[0] not in first_person and right[0] not in first_person
    return False


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
    sample_duration: float = 1.0,
    *,
    stabilize_text: bool = True,
) -> list[Cue]:
    ordered = sorted(detections)
    stable = list(ordered)
    frame_gap = min(max_gap, sample_duration * 1.5)
    if stabilize_text:
        for index in range(1, len(ordered) - 1):
            before, middle, after = ordered[index - 1:index + 2]
            if not (0 < middle[0] - before[0] <= frame_gap and 0 < after[0] - middle[0] <= frame_gap):
                continue
            first_words, middle_words, last_words = [_caption_words(text) for _, text in (before, middle, after)]
            if (min(_spoken_word_count(first_words), _spoken_word_count(middle_words)) < 5
                    or first_words != last_words):
                continue
            if (not _protected_caption_change(before[1], middle[1])
                    and _single_character_change("".join(first_words), "".join(middle_words))):
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
            fragment = (stabilize_text and _spoken_word_count(short.split()) >= 3
                        and not short.endswith((".", "?", "!"))
                        and short in long and len(short) >= len(long) * 0.55
                        and [index for index, word in enumerate(left_words) if word == "_"]
                        == [index for index, word in enumerate(right_words) if word == "_"])
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


def _vision_frames(binary: str, frames: list[Path], caption_region: bool = False) -> dict[str, dict]:
    with compute_phase("vision"):
        return _run_vision_frames(binary, frames, caption_region)


def _run_vision_frames(binary: str, frames: list[Path], caption_region: bool = False) -> dict[str, dict]:
    lease = compute_lease_fd()
    res = subprocess.run([binary, "--json"] + (["--caption-region"] if caption_region else [])
                         + [str(frame) for frame in frames],
             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
             check=False, timeout=max(120, len(frames) * 10),
             pass_fds=() if lease is None else (lease,))
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
        by_file[name] = row
    if set(by_file) != {frame.name for frame in frames}:
        raise RuntimeError("Vision OCR returned unexpected caption frames")
    return by_file


def _scan_caption_frames(
    video: str | Path,
    gaps: list[tuple[float, float]],
    ocr_bin: str | None = None,
    fps: float = 2.0,
    tmp_dir: Path | None = None,
    ignore_patterns: Iterable[re.Pattern] = DEFAULT_IGNORE_PATTERNS,
    progress: Callable[[int, int], None] | None = None,
    center_tolerance: float | None = 0.12,
    *,
    retry_all: bool = False,
    frame_sink: Callable[[Path, float, dict, dict | None], None] | None = None,
) -> list[tuple[float, str]]:
    if not gaps:
        return []

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("OCR frame rate must be positive")
    binary = ocr_bin or get_vision_ocr_bin()
    if not binary:
        raise RuntimeError(
            "Apple Vision OCR is unavailable. On macOS, install the Swift compiler for the bundled tool "
            "or set SUBZERO_VISION_OCR to an existing executable."
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
            # Millisecond PTS can subtract exact 100 ms intervals to just below 0.1.
            cmd = [
                "ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", f"{g_start:.3f}", "-i", str(video),
                "-t", f"{dur:.3f}", "-an", "-sn", "-vf",
                f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{1 / fps:.9f}-0.000000001),showinfo",
                "-fps_mode", "vfr", "-q:v", "2",
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
                by_file = _vision_frames(binary, frames)
                ordered = [by_file[frame.name] for frame in frames]
                selected = ([index for index, frame in enumerate(ordered) if not _caption_title_card(frame)]
                            if retry_all else caption_retry_indices(ordered, timestamps, fps))
                retried = {}
                if selected:
                    retry_frames = [frames[index] for index in selected]
                    retried = _vision_frames(binary, retry_frames, caption_region=True)
                    readings = recover_caption_runs(ordered, {index: retried[frames[index].name] for index in selected},
                                                    timestamps, center_tolerance)
                else:
                    readings = None
                detections = []
                for position, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                    if timestamp >= dur:
                        continue
                    previous = (by_file[frames[position - 1].name] if position > 0
                                and 0 < timestamp - timestamps[position - 1] <= 1.5 / fps else None)
                    following = (by_file[frames[position + 1].name] if position + 1 < len(frames)
                                 and 0 < timestamps[position + 1] - timestamp <= 1.5 / fps else None)
                    text = (readings[position] if readings is not None else
                            caption_text(by_file[frame.name], center_tolerance, previous=previous, following=following))
                    text = clean_ocr_text(text, ignore_patterns)
                    detections.append((g_start + timestamp, text))
                    if frame_sink is not None:
                        frame_sink(frame, g_start + timestamp, {**by_file[frame.name], "acceptedText": text},
                                   retried.get(frame.name))
                recovered.extend(detections)
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
    detections = _scan_caption_frames(video, gaps, ocr_bin, fps, tmp_dir, ignore_patterns,
                                      progress, center_tolerance)
    recovered = []
    for start, end in gaps:
        frames = [(timestamp, text) for timestamp, text in detections if start <= timestamp < end]
        cues = cluster_ocr_detections(frames, min_duration=0, max_gap=1.5 / fps, sample_duration=1 / fps)
        recovered.extend(clip_cues(cues, [(start, end)]))
    return recovered


def extract_all_captions(video: str | Path, duration: float, *,
                         progress: Callable[[int, int], None] | None = None,
                         cache_dir: str | Path | None = None,
                         caption_rescue: CaptionRescue | None = None) -> list[Cue]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Caption scanning requires a positive video duration")
    detections = []
    cache = _caption_scan_cache(video, Path(cache_dir)) if cache_dir is not None else None
    starts = range(0, math.ceil(duration), 120)
    for index, start in enumerate(starts):
        end = min(duration, start + 120)
        windows = [(max(0, start - 1), min(duration, end + 1))]
        frames = cache.read(windows, fps=2) if cache is not None else None
        if frames is None:
            frames = _scan_caption_frames(video, windows)
            if cache is not None:
                cache.write(windows, frames, fps=2)
        detections.extend((timestamp, text) for timestamp, text in frames if start <= timestamp < end)
        if progress:
            progress(int(50 * (index + 1) / len(starts)), 100)
    options = {"caption_rescue": caption_rescue} if caption_rescue is not None else {}
    return refine_caption_timing(video, detections, duration, progress=progress, scan_cache=cache, **options)


def caption_transition_windows(detections: list[tuple[float, str]], duration: float) -> list[tuple[float, float]]:
    windows = []
    previous_words = []
    previous_time = None
    for timestamp, text in sorted(detections) + [(duration, "")]:
        words = _caption_words(text)
        if words != previous_words:
            before = previous_time if previous_time is not None else timestamp - 0.5
            start = max(0, timestamp - 0.75, min(before, timestamp - 0.5))
            end = min(duration, timestamp + 0.5)
            if windows and start <= windows[-1][1] + 0.1:
                windows[-1] = (windows[-1][0], end)
            elif end > start:
                windows.append((start, end))
        previous_words, previous_time = words, timestamp
    return [(round(start, 3), round(end, 3)) for start, end in windows]


def _stabilize_dense_readings(detections: list[tuple[float, str]]) -> list[tuple[float, str]]:
    stable = list(detections)
    words = [_caption_words(text) for _, text in detections]
    for left, (start, caption) in enumerate(detections):
        expected = words[left]
        if _spoken_word_count(expected) < 5:
            continue
        lines = [_caption_words(line) for line in caption.splitlines()]
        for right in range(left + 1, len(detections)):
            timestamp = detections[right][0]
            if (timestamp - start > 0.501 or not words[right]
                    or not 0 < timestamp - detections[right - 1][0] <= 0.151):
                break
            if words[right] != expected:
                continue
            between = range(left + 1, right)
            if all(words[index] == expected or (_spoken_word_count(words[index]) >= 3 and (
                    words[index] in lines or (
                        right == left + 2 and len(expected) == len(words[index]) + 1
                        and not _protected_caption_change(detections[index][1], caption)
                        and _single_character_change("".join(words[index]), "".join(expected)))))
                    for index in between):
                for index in between:
                    stable[index] = (detections[index][0], caption)
    return stable


def _normalize_english_caption(text: str) -> str:
    text = re.sub(r"(?<!\w)([Ii]|[Yy]ou|[Hh]e|[Ss]he|[Ii]t|[Ww]e|[Tt]hey)(['’])[Il]{2}(?!\w)",
                  lambda match: match[1] + match[2] + "ll", text)
    return re.sub(r"(?<!\w)(ain|aren|can|couldn|daren|didn|doesn|don|hadn|hasn|haven|isn|mightn|mustn|"
                  r"needn|oughtn|shan|shouldn|wasn|weren|won|wouldn)[ \t]+t(?![\w-])",
                  lambda match: match[1] + "'t", text)


def refine_caption_timing(video: str | Path, detections: list[tuple[float, str]], duration: float, *,
                          progress: Callable[[int, int], None] | None = None,
                          scan_cache: CaptionScanCache | None = None,
                          caption_rescue: CaptionRescue | None = None) -> list[Cue]:
    windows = caption_transition_windows(detections, duration)
    if not windows:
        if progress:
            progress(100, 100)
        return []

    def dense_progress(done: int, total: int):
        if progress:
            extent = 40 if caption_rescue is not None else 50
            progress(50 + int(extent * done / max(1, total)), 100)

    if scan_cache is None:
        dense = _scan_caption_frames(video, windows, fps=10, retry_all=True, progress=dense_progress)
    else:
        dense = []
        for index, window in enumerate(windows, start=1):
            frames = scan_cache.read([window], fps=10, retry_all=True)
            if frames is None:
                frames = _scan_caption_frames(video, [window], fps=10, retry_all=True)
                scan_cache.write([window], frames, fps=10, retry_all=True)
            dense.extend(frames)
            dense_progress(index, len(windows))
    losses = [] if caption_rescue is not None else None
    cues = _interpret_caption_frames(detections, dense, windows, duration, losses=losses)
    if caption_rescue is not None:
        intervals = _caption_rescue_windows(detections, cues, losses, duration)
        if intervals:
            options = {"progress": lambda done, total: progress(90 + int(9 * done / max(1, total)), 100)} if progress else {}
            detections, dense = _rescue_caption_frames(video, detections, dense, duration, windows,
                                                       intervals, caption_rescue, **options)
            cues = _interpret_caption_frames(detections, dense, windows, duration)
        validate_caption_readings(cues)
        if progress:
            progress(100, 100)
    return cues


def _caption_rescue_windows(detections, cues, losses, duration):
    intervals = list(losses)
    for start in range(len(cues)):
        for end in range(start + 3, min(start + 8, len(cues)) + 1):
            try:
                validate_caption_readings(cues[start:end])
            except RuntimeError:
                first = start
                while first + 3 < end:
                    try:
                        validate_caption_readings(cues[first + 1:end])
                    except RuntimeError:
                        first += 1
                    else:
                        break
                stamp = TIME.search(f"{cues[end - 1].start} --> {cues[end - 1].end}")
                intervals.append((cue_start_seconds(cues[first]), _to_seconds(*stamp.groups()[4:])))
                break
    if not intervals:
        return []
    runs = cluster_ocr_detections(detections, min_duration=0, max_gap=.75, sample_duration=.5)
    expanded = []
    for left, right in intervals:
        for cue in runs:
            stamp = TIME.search(f"{cue.start} --> {cue.end}")
            start, end = _to_seconds(*stamp.groups()[:4]), _to_seconds(*stamp.groups()[4:])
            if start < right and end > left:
                left, right = min(left, start), max(right, end)
        expanded.append((max(0, left - .75), min(duration, right + .75)))
    merged = []
    for left, right in sorted(expanded):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _capture_caption_frame(source: Path, timestamp: float, full: dict, roi: dict | None,
                           directory: Path) -> CaptionFrame:
    raw = _image_bytes(source)
    source_digest = hashlib.sha256(raw).hexdigest()
    admitted = not _caption_title_card(full) and bool(
        _caption_regions(full, .12, max_height=.11)
        or (roi is not None and _caption_regions(roi, .12, max_height=.11)))
    prefix = f"{timestamp:.6f}-{source_digest[:16]}"
    image = directory / f"{prefix}{source.suffix}"
    image.write_bytes(raw)
    if admitted:
        image = directory / f"{prefix}.png"
        image.write_bytes(crop_caption_image(raw))
    image_digest = hashlib.sha256(_image_bytes(image)).hexdigest()
    evidence = {"source_sha256": source_digest, "admitted": admitted, "admission_version": CAPTION_SCAN_VERSION,
                "crop": CAPTION_CROP if admitted else None,
                "native_full": {key: value for key, value in full.items() if key != "file"},
                "native_roi": {key: value for key, value in roi.items() if key != "file"} if roi else None}
    return CaptionFrame(round(timestamp, 3), image, image_digest, evidence)


def _rescue_caption_frames(video, coarse, dense, duration, dense_windows, intervals, client, *, progress=None):
    selected = {round(timestamp, 3) for timestamp, _ in coarse + dense
                if any(start <= timestamp < end for start, end in intervals)}
    before = fingerprint(video)
    with tempfile.TemporaryDirectory(prefix="subzero_caption_rescue_") as temporary:
        captured = {}

        def capture(source, timestamp, full, roi):
            key = round(timestamp, 3)
            if key not in selected:
                return
            frame = _capture_caption_frame(source, timestamp, full, roi, Path(temporary))
            previous = captured.get(key)
            if previous is not None:
                if (previous.image_digest != frame.image_digest
                        or previous.evidence["admitted"] != frame.evidence["admitted"]):
                    raise RuntimeError(f"Caption replay produced different evidence at exact frame {key:.3f}s")
            else:
                captured[key] = frame

        requests = []
        for start in range(0, math.ceil(duration), 120):
            end = min(duration, start + 120)
            if any(left < end and right > start for left, right in intervals):
                requests.append(((max(0, start - 1), min(duration, end + 1)), 2))
        for start, end in dense_windows:
            if any(left < end and right > start for left, right in intervals):
                requests.append(((start, end), 10))
        total = len(requests) + len(selected)
        for index, (window, fps) in enumerate(requests):
            if progress:
                progress(index, total)
            _scan_caption_frames(video, [window], fps=fps, retry_all=True, frame_sink=capture)
        if progress:
            progress(len(requests), total)
        if set(captured) != selected:
            raise RuntimeError("Caption replay did not capture every exact input frame")
        if fingerprint(video) != before:
            raise RuntimeError("Video changed during caption frame capture")
        options = {"progress": lambda done, count: progress(len(requests) + done, total)} if progress else {}
        readings = client.read_frames(before, [captured[key] for key in sorted(captured)], **options)
        if set(readings) != selected or not all(isinstance(text, str) for text in readings.values()):
            raise RuntimeError("Caption rescue did not return every exact input frame")
        replacements = {key: clean_ocr_text(readings[key]) if frame.evidence["admitted"] else ""
                        for key, frame in captured.items()}
        for cue in cluster_ocr_detections(coarse, min_duration=0, max_gap=.75, sample_duration=.5):
            stamp = TIME.search(f"{cue.start} --> {cue.end}")
            start, end = _to_seconds(*stamp.groups()[:4]), _to_seconds(*stamp.groups()[4:])
            support = [round(timestamp, 3) for timestamp, text in coarse if start <= timestamp < end
                       and _caption_words(text) == _caption_words(cue.text)]
            if len(support) >= 2 and any(key in replacements and not replacements[key] for key in support):
                raise RuntimeError(f"Caption rescue lost a confirmed coarse caption near {start:.3f}s")
        if fingerprint(video) != before:
            raise RuntimeError("Video changed during caption recognition")
        return ([(timestamp, replacements.get(round(timestamp, 3), text)) for timestamp, text in coarse],
                [(timestamp, replacements.get(round(timestamp, 3), text)) for timestamp, text in dense])


def _interpret_caption_frames(detections, dense, windows, duration, *, losses=None):
    detections = [(timestamp, _normalize_english_caption(text)) for timestamp, text in detections]
    dense = _stabilize_dense_readings([(timestamp, _normalize_english_caption(text)) for timestamp, text in dense])
    anchors = []
    for cue in cluster_ocr_detections(detections, min_duration=0, max_gap=0.75, sample_duration=0.5):
        stamp = TIME.search(f"{cue.start} --> {cue.end}")
        start, end = _to_seconds(*stamp.groups()[:4]), _to_seconds(*stamp.groups()[4:])
        words = _caption_words(cue.text)
        support = [timestamp for timestamp, text in detections
                   if start - 0.001 <= timestamp < end and _caption_words(text) == words]
        variants, readings = Counter(), {}
        for timestamp, text in dense:
            if not start <= timestamp < end:
                continue
            observed = tuple(_caption_words(text))
            if observed == tuple(words) or ((_spoken_word_count(observed) >= 5 or _trailing_caption_glyph(cue.text, text))
                    and not _protected_caption_change(cue.text, text)
                    and _single_character_change("".join(words), "".join(observed))):
                variants[observed] += 1
                readings.setdefault(observed, text)
        ranked = variants.most_common(2)
        confirmed = bool(ranked and ranked[0][1] >= 2
                         and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]))
        caption = readings[ranked[0][0]] if confirmed else cue.text
        anchors.append((start, end, caption, support, confirmed))
    stabilized = []
    for timestamp, text in dense:
        words = _caption_words(text)
        matches = []
        for start, end, caption, support, confirmed in anchors:
            if not words or not start - 0.75 <= timestamp <= end + 0.75:
                continue
            expected = _caption_words(caption)
            exact = words == expected
            partial_line = (_spoken_word_count(words) >= 3 and support and min(support) <= timestamp <= max(support)
                            and any(words == _caption_words(line) for line in caption.splitlines()))
            flicker = (confirmed and (_spoken_word_count(expected) >= 5 or _trailing_caption_glyph(text, caption))
                       and not _protected_caption_change(text, caption)
                       and _single_character_change("".join(words), "".join(expected)))
            if exact or (len(support) >= 2 and (partial_line or flicker)):
                matches.append((int(exact), start <= timestamp <= end,
                                -abs(timestamp - (start + end) / 2), caption))
        stabilized.append((timestamp, max(matches)[-1] if matches else text))
    stabilized = _stabilize_dense_readings(stabilized)
    combined = {round(timestamp, 3): text for timestamp, text in detections
                if not any(start <= timestamp < end for start, end in windows)}
    combined.update((round(timestamp, 3), text) for timestamp, text in stabilized if 0 <= timestamp < duration)
    for start, end, caption, support, _ in anchors:
        if len(support) >= 2 and not any(start - 0.75 <= timestamp <= end + 0.75
                                        and _caption_words(text) == _caption_words(caption)
                                        for timestamp, text in combined.items()):
            if losses is None:
                raise RuntimeError(f"Dense caption verification lost a confirmed caption near {start:.3f}s")
            losses.append((start, end))
    combined[duration] = ""
    edges = []
    previous = None
    for timestamp, text in sorted(combined.items()):
        boundary = timestamp
        if (previous is not None and timestamp != duration and 0 < timestamp - previous[0] <= 0.151
                and _caption_words(text) != _caption_words(previous[1])):
            boundary = (timestamp + previous[0]) / 2
        edges.append((boundary, text))
        previous = timestamp, text
    return cluster_ocr_detections(edges, min_duration=0, max_gap=0.75, sample_duration=0.5,
                                  stabilize_text=False)


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
    all_captions: bool = False,
    caption_cache_dir: str | Path | None = None,
    caption_rescue: CaptionRescue | None = None,
    translation_client=None,
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
    if all_captions:
        ref = build_reference(video, cache_dir or Path.home() / ".cache" / "subzero" / "references")
        duration = float(ref.get("duration") or 0)
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("Video duration is missing from the caption reference")
        gaps = [(0, duration)]
    else:
        gaps = find_caption_gaps(video, content, cache_dir=cache_dir)
    total_speech_sec = sum(end - start for start, end in gaps)

    if not gaps:
        return GapReport(total_gaps=0, speech_seconds=0.0, cues_recovered=0, cues=[])

    scan_options = {"cache_dir": caption_cache_dir} if caption_cache_dir is not None else {}
    if caption_rescue is not None:
        if not all_captions:
            raise ValueError("Caption rescue requires whole-video caption recovery")
        scan_options["caption_rescue"] = caption_rescue
    recovered = (extract_all_captions(video, duration, progress=progress, **scan_options) if all_captions else
                 extract_and_ocr_gaps(video, gaps, progress=progress))
    if not recovered:
        return GapReport(total_gaps=len(gaps), speech_seconds=total_speech_sec, cues_recovered=0, cues=[])

    to_merge = recovered
    if target_lang and target_lang.lower() not in ("en", "eng"):
        validate_caption_readings(recovered)
        if translation_client is not None:
            client = translation_client
        elif provider.lower() in ("openai", "openrouter", "groq", "deepseek"):
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

    if all_captions:
        from .caption_timeline import merge_caption_sources
        combined = merge_caption_sources(existing_cues, to_merge)
        originals = Counter(existing_cues)
        to_merge = []
        for cue in combined:
            if originals[cue]:
                originals[cue] -= 1
            else:
                to_merge.append(cue)
    else:
        to_merge = clip_cues(to_merge, gaps)
        to_merge = parse_srt(fix_text(dump_srt(to_merge), Options(max_line=42, preserve_breaks=False)).text)
        combined = sorted(existing_cues + to_merge, key=cue_start_seconds)

    out_path = Path(output) if output else sub_p
    if not dry_run and to_merge:
        if out_path.is_symlink():
            raise RuntimeError("Refusing to replace a subtitle symlink")
        rendered = dump_srt(combined)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=out_path.parent,
                                         prefix=".subtitle-", suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            try:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            except (OSError, UnicodeError):
                handle.close()
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
