"""Reject rapid OCR word oscillations without choosing or rewriting a reading."""

from __future__ import annotations

from dataclasses import dataclass
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
