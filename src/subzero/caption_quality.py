"""Reject rapid OCR word oscillations without choosing or rewriting a reading."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
import html
import re
import unicodedata

from .convert import Cue

_SPEAKER = re.compile(r"(?m)^\s*(?:-\s*)?([A-Z][A-Z0-9 '\-]{0,39}):\s*")


@dataclass(frozen=True)
class _Reading:
    start: int
    end: int
    text: str
    speakers: frozenset[str]


def _milliseconds(stamp: str) -> int:
    hours, minutes, seconds = stamp.split(':')
    seconds, millis = seconds.split(',')
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)


def _reading(cue: Cue) -> _Reading:
    text = html.unescape(re.sub(r'<[^>]*>|\{\\[^}]*\}', '', cue.text))
    text = text.replace(r'\N', '\n').replace(r'\n', '\n')
    speakers = frozenset(_SPEAKER.findall(text))
    text = unicodedata.normalize('NFKC', _SPEAKER.sub('', text)).casefold()
    text = re.sub(r'_+', '_', text)
    text = ''.join(character for character in text if character.isalnum() or character == '_')
    return _Reading(_milliseconds(cue.start), _milliseconds(cue.end), text, speakers)


def _similar(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right or max(len(left), len(right)) > 500:
        return False
    if left.replace('_', '') == right.replace('_', ''):
        return True
    limit = 1 if max(len(left), len(right)) < 8 else max(2, min(4, min(len(left), len(right)) // 15))
    if abs(len(left) - len(right)) > limit:
        return False
    edits = sum(max(b - a, d - c) for kind, a, b, c, d in
                SequenceMatcher(None, left, right, autojunk=False).get_opcodes() if kind != 'equal')
    return edits <= limit


def stabilize_caption_readings(cues: list[Cue]) -> list[Cue]:
    """Collapse an A/B/A flicker run onto its dominant reading.

    A frozen caption can be read with a one-glyph difference on individual
    frames. validate_caption_readings rejects the whole recovery for that;
    when the variants stay similar, contiguous, short, and one holds a strict
    majority of the run's screen time, the run is rewritten to that reading.
    Ties, sustained changes, and differing censorship-bar layouts pass through
    untouched and still reach review.
    """
    ordered = sorted(((_reading(cue), cue) for cue in cues),
                     key=lambda pair: (pair[0].start, pair[0].end))
    replaced = False
    index = 0
    while index < len(ordered):
        first, _ = ordered[index]
        speakers = first.speakers
        short = 0 < first.end - first.start < 300
        states = [first.text]
        window = [index]
        cursor = index + 1
        while cursor < len(ordered) and len(window) < 64:
            current, cue = ordered[cursor]
            previous, _ = ordered[cursor - 1]
            if (not 0 <= current.start - previous.end <= 200
                    or len(speakers | current.speakers) > 1
                    or not _similar(first.text, current.text)):
                break
            speakers |= current.speakers
            short = short or 0 < current.end - current.start < 300
            if current.text != states[-1]:
                states.append(current.text)
            window.append(cursor)
            cursor += 1
        returned = any(states[position] in states[:position] for position in range(2, len(states)))
        bars = {tuple(slot for slot, char in enumerate(ordered[position][0].text) if char == '_')
                for position in window}
        if short and returned and len(bars) == 1:
            weights = Counter()
            for position in window:
                reading = ordered[position][0]
                weights[reading.text] += reading.end - reading.start
            dominant, weight = weights.most_common(1)[0]
            if weight > 0 and weight * 2 > sum(weights.values()):
                raw = next(ordered[position][1].text for position in window
                           if ordered[position][0].text == dominant)
                for position in window:
                    reading, cue = ordered[position]
                    if cue.text != raw:
                        ordered[position] = (reading, replace(cue, text=raw))
                        replaced = True
        index = cursor
    if not replaced:
        return cues
    return [cue for _, cue in ordered]


def validate_caption_readings(cues: list[Cue]) -> None:
    """Flag brief A/B/A word variants; timing and semantic correctness need other checks."""
    readings = sorted((_reading(cue) for cue in cues), key=lambda reading: (reading.start, reading.end))
    for index, first in enumerate(readings):
        previous = first
        states = [first.text]
        speakers = first.speakers
        short = 0 < first.end - first.start < 300
        changed_at = None
        # Eight consecutive readings bound comparison cost even in malformed dense inputs.
        for current in readings[index + 1:index + 8]:
            if (not 0 <= current.start - previous.end <= 200
                    or len(speakers | current.speakers) > 1
                    or not _similar(first.text, current.text)):
                break
            speakers |= current.speakers
            short |= 0 < current.end - current.start < 300
            if current.text != states[-1]:
                if changed_at is None:
                    changed_at = current.start
                if current.start - changed_at > 3000:
                    break
                if current.text in states[:-1] and short:
                    raise RuntimeError(f'Unstable OCR caption readings near {changed_at / 1000:.3f} seconds')
                states.append(current.text)
            previous = current
