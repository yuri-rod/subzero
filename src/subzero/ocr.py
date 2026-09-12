"""Speech gap analysis and burned-in open caption OCR recovery."""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
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
    re.compile(r"^(tuku|lavo|gata|vatu|civa|medical|survivor)\b", re.I),
    re.compile(r"^[A-Z\s]{1,15}\bDAY\s+\d+", re.I),
)


def get_vision_ocr_bin() -> str | None:
    env_bin = os.getenv("SUBZERO_VISION_OCR")
    if env_bin and Path(env_bin).is_file() and os.access(env_bin, os.X_OK):
        return env_bin

    which_bin = shutil.which("vision_ocr")
    if which_bin:
        return which_bin

    repo_bin = Path(__file__).resolve().parent.parent.parent / "tools" / "vision_ocr"
    if repo_bin.is_file() and os.access(repo_bin, os.X_OK):
        return str(repo_bin)

    swift_src = Path(__file__).resolve().parent.parent.parent / "tools" / "vision_ocr.swift"
    if not swift_src.is_file():
        swift_src = Path(__file__).resolve().parent / "vision_ocr.swift"

    if swift_src.is_file() and os.uname().sysname == "Darwin":
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
    speech = ref.get("speech", [])
    sub_spans = spans(subtitle_content)

    gaps: list[tuple[float, float]] = []
    for sp_start, sp_end in speech:
        dur = sp_end - sp_start
        if dur < min_duration:
            continue
        overlap = 0.0
        for sub_start, sub_end in sub_spans:
            if sub_end <= sp_start:
                continue
            if sub_start >= sp_end:
                break
            overlap += max(0.0, min(sp_end, sub_end) - max(sp_start, sub_start))
        if (overlap / dur) <= max_coverage:
            gaps.append((round(sp_start, 3), round(sp_end, 3)))
    return gaps


def clean_ocr_text(
    text: str,
    ignore_patterns: Iterable[re.Pattern] = DEFAULT_IGNORE_PATTERNS,
) -> str:
    text = re.sub(r"[ạẠỊ]", "", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" \t\r\n'\"`´‘“”)({}[]").strip()
        if not line or len(line) < 3:
            continue
        words = line.split()
        if len(words) >= 2 and line.isupper():
            continue
        if any(pat.search(line) for pat in ignore_patterns):
            continue
        lines.append(line)
    return "\n".join(lines)


def fmt_srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def cluster_ocr_detections(
    detections: list[tuple[float, str]],
    min_duration: float = 1.4,
    max_gap: float = 2.0,
    fuzzy_ratio: float = 0.60,
) -> list[Cue]:
    if not detections:
        return []

    merged: list[tuple[float, float, str]] = []
    curr_start, curr_text = detections[0]
    curr_end = curr_start + 1.2

    for t, txt in detections[1:]:
        gap = t - curr_end
        ratio = difflib.SequenceMatcher(None, curr_text.lower(), txt.lower()).ratio()
        is_match = (
            txt == curr_text
            or txt in curr_text
            or curr_text in txt
            or ratio >= fuzzy_ratio
        )
        if is_match and gap < max_gap:
            curr_end = max(curr_end, t + 1.2)
            if len(txt) > len(curr_text):
                curr_text = txt
        else:
            if (curr_end - curr_start) < min_duration:
                curr_end = curr_start + min_duration
            merged.append((curr_start, curr_end, curr_text))
            curr_start = t
            curr_end = t + 1.2
            curr_text = txt

    if (curr_end - curr_start) < min_duration:
        curr_end = curr_start + min_duration
    merged.append((curr_start, curr_end, curr_text))

    return [
        Cue(start=fmt_srt_time(s), end=fmt_srt_time(e), text=t)
        for s, e, t in merged
    ]


def extract_and_ocr_gaps(
    video: str | Path,
    gaps: list[tuple[float, float]],
    ocr_bin: str | None = None,
    fps: float = 1.0,
    tmp_dir: Path | None = None,
    ignore_patterns: Iterable[re.Pattern] = DEFAULT_IGNORE_PATTERNS,
) -> list[Cue]:
    if not gaps:
        return []

    binary = ocr_bin or get_vision_ocr_bin()
    if not binary:
        raise RuntimeError(
            "vision_ocr tool not found. Compile tools/vision_ocr.swift or set SUBZERO_VISION_OCR."
        )

    detections: list[tuple[float, str]] = []
    cleanup_tmp = tmp_dir is None
    work_dir = tmp_dir or Path(tempfile.mkdtemp(prefix="subzero_ocr_"))

    try:
        for idx, (g_start, g_end) in enumerate(gaps, start=1):
            gap_dir = work_dir / f"gap_{idx:03d}"
            gap_dir.mkdir(parents=True, exist_ok=True)
            dur = max(0.5, g_end - g_start)
            cmd = [
                "ffmpeg", "-y", "-ss", f"{g_start:.3f}", "-i", str(video),
                "-t", f"{dur:.3f}", "-vf", f"fps={fps}", "-q:v", "2",
                str(gap_dir / "f_%03d.jpg"),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            frames = sorted(gap_dir.glob("*.jpg"))
            if not frames:
                continue

            res = subprocess.run(
                [binary, "--json"] + [str(f) for f in frames],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            try:
                data = json.loads(res.stdout)
            except Exception:
                data = []

            for f_idx, item in enumerate(data):
                raw_text = item.get("subtitleText", "")
                cleaned = clean_ocr_text(raw_text, ignore_patterns=ignore_patterns)
                if cleaned:
                    t = g_start + (f_idx / fps)
                    detections.append((t, cleaned))
    finally:
        if cleanup_tmp and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    return cluster_ocr_detections(detections)


def cue_start_seconds(c: Cue) -> float:
    m = TIME.search(f"{c.start} --> {c.end}")
    if not m:
        return 0.0
    return _to_seconds(*m.groups()[:4])


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
    content, _ = read(sub_p)
    gaps = find_speech_gaps(video, content, cache_dir=cache_dir)
    total_speech_sec = sum(end - start for start, end in gaps)

    if not gaps:
        return GapReport(total_gaps=0, speech_seconds=0.0, cues_recovered=0, cues=[])

    recovered = extract_and_ocr_gaps(video, gaps)
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
            chosen_model = model or "translategemma:4b"
            client = OllamaClient(url=chosen_url, model=chosen_model)

        to_merge = translate_cues(
            recovered,
            target_lang=target_lang,
            client=client,
            source_lang="en",
            progress=progress,
        )

    out_path = Path(output) if output else sub_p
    if not dry_run:
        if backup and out_path.resolve() == sub_p.resolve():
            bak = sub_p.with_suffix(".srt.bak")
            if not bak.exists():
                bak.write_text(content, encoding="utf-8")

        existing_cues = parse_srt(content)
        all_cues = existing_cues + to_merge
        all_cues.sort(key=cue_start_seconds)

        rendered = dump_srt(all_cues)
        try:
            from .worker.guards import sanitize_to_excellence
            sanitized = sanitize_to_excellence(rendered, target_lang=target_lang or "pt-BR")
        except Exception:
            sanitized = rendered
        fixed = fix_text(sanitized, Options(max_line=42, preserve_breaks=False))
        out_path.write_text(fixed.text, encoding="utf-8", errors="replace")

    return GapReport(
        total_gaps=len(gaps),
        speech_seconds=total_speech_sec,
        cues_recovered=len(to_merge),
        cues=to_merge,
        target=str(out_path),
    )
