"""Merge and align dual-language (bilingual) subtitles."""

from __future__ import annotations

from pathlib import Path

from .convert import Cue, parse_srt, dump_srt
from .core import read


def _time_to_ms(time_str: str) -> int:
    time_str = time_str.strip().replace(".", ",")
    h, m, rest = time_str.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


def merge_cues(
    primary_cues: list[Cue],
    secondary_cues: list[Cue],
    separator: str = "\n",
    secondary_color: str | None = None,
) -> list[Cue]:
    """Merge primary and secondary subtitle cues by overlapping timeframes."""
    if not primary_cues:
        return secondary_cues
    if not secondary_cues:
        return primary_cues

    merged: list[Cue] = []
    sec_idx = 0
    num_sec = len(secondary_cues)

    for p_cue in primary_cues:
        p_start = _time_to_ms(p_cue.start)
        p_end = _time_to_ms(p_cue.end)

        while sec_idx < num_sec and _time_to_ms(secondary_cues[sec_idx].end) <= p_start:
            sec_idx += 1

        matched_texts: list[str] = []
        curr = sec_idx
        while curr < num_sec:
            s_cue = secondary_cues[curr]
            s_start = _time_to_ms(s_cue.start)
            s_end = _time_to_ms(s_cue.end)

            if s_start >= p_end:
                break

            overlap = min(p_end, s_end) - max(p_start, s_start)
            if overlap > 200 or (overlap > 0 and overlap >= (s_end - s_start) * 0.3):
                if s_cue.text and s_cue.text not in matched_texts:
                    matched_texts.append(s_cue.text)
            curr += 1

        if matched_texts:
            sec_combined = "\n".join(matched_texts)
            if secondary_color:
                sec_combined = f'<font color="{secondary_color}">{sec_combined}</font>'
            combined_text = f"{p_cue.text}{separator}{sec_combined}"
        else:
            combined_text = p_cue.text

        merged.append(Cue(start=p_cue.start, end=p_cue.end, text=combined_text))

    return merged


def merge_files(
    primary_path: str | Path,
    secondary_path: str | Path,
    output: str | Path | None = None,
    separator: str = "\n",
    secondary_color: str | None = None,
) -> tuple[Path, int]:
    """Merge two subtitle files into a single bilingual subtitle file."""
    p1 = Path(primary_path)
    p2 = Path(secondary_path)

    raw1, _ = read(p1)
    raw2, _ = read(p2)

    cues1 = parse_srt(raw1)
    cues2 = parse_srt(raw2)

    if not cues1 and not cues2:
        raise ValueError(f"No cues found in {p1} or {p2}")

    merged = merge_cues(cues1, cues2, separator=separator, secondary_color=secondary_color)

    out_path = Path(output) if output else p1.parent / f"{p1.stem}.bilingual.srt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_srt(merged), encoding="utf-8")
    return out_path, len(merged)
