"""Keep overlapping dialogue independent until its final display composition."""

from __future__ import annotations

import html
import re
from dataclasses import replace

from .convert import Cue

_STAMP = re.compile(r'(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})')
_SPEAKER = r"[A-Z][A-Z0-9 '\-]{1,40}:\s*"
_TURN = re.compile(r'(?m)^\s*[-\u2013\u2014]\s*')


def _milliseconds(stamp: str) -> int:
    match = _STAMP.fullmatch(stamp)
    if match is None:
        raise ValueError(f'Invalid caption timestamp: {stamp!r}')
    hours, minutes, seconds, millis = map(int, match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _timestamp(millis: int) -> str:
    seconds, millis = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}'


def _ordered(cues: list[Cue]) -> list[Cue]:
    for cue in cues:
        if _milliseconds(cue.end) <= _milliseconds(cue.start) or not cue.text.strip():
            raise ValueError(f'Invalid caption interval or empty text at {cue.start}')
    return sorted(cues, key=lambda cue: (_milliseconds(cue.start), _milliseconds(cue.end)))


def _plain(text: str) -> str:
    text = re.sub(r'\{\\[^}]*\}', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    return html.unescape(text).replace(r'\N', '\n').replace(r'\n', '\n').replace(r'\h', ' ')


def _words(text: str) -> tuple[str, ...]:
    text = _TURN.sub('', _plain(text))
    text = re.sub(r'(?m)^\s*' + _SPEAKER, '', text)
    return tuple(re.findall(r"\w+(?:'\w+)*", text.replace('’', "'").casefold()))


def _voices(text: str) -> set[tuple[str, ...]]:
    text = _plain(text)
    pieces = re.split(r'(?m)^\s*(?:[-\u2013\u2014]\s*|' + _SPEAKER + ')', text)
    return {_words(text), *(_words(piece) for piece in pieces if piece.strip())}


def _speakers(text: str) -> set[str]:
    return {match.group().split(':', 1)[0].strip() for match in
            re.finditer(r'(?m)^\s*' + _SPEAKER, _TURN.sub('', _plain(text)))}


def _same_speaker(first: str, second: str) -> bool:
    left, right = _speakers(first), _speakers(second)
    return not left or not right or bool(left & right)


def _overlap_enough(start: int, end: int, other_start: int, other_end: int) -> bool:
    overlap = min(end, other_end) - max(start, other_start)
    return overlap > 0 and overlap * 2 >= min(end - start, other_end - other_start)


def _already_original(cue: Cue, originals: list[Cue]) -> bool:
    words = _words(cue.text)
    if not words:
        return False
    start, end = _milliseconds(cue.start), _milliseconds(cue.end)
    for index, original in enumerate(originals):
        first, last = _milliseconds(original.start), _milliseconds(original.end)
        if first >= end:
            break
        if last <= start:
            continue
        if len(_speakers(original.text)) > 1:
            continue
        if not _same_speaker(cue.text, original.text):
            continue
        if words in _voices(original.text) and _overlap_enough(start, end, first, last):
            return True
        joined = _words(original.text)
        speakers = _speakers(original.text)
        for following in originals[index + 1:index + 8]:
            next_start, next_end = _milliseconds(following.start), _milliseconds(following.end)
            if not 0 <= next_start - last <= 500 or next_start >= end or next_end - first > 20000:
                break
            if words[:len(joined)] != joined:
                break
            speakers |= _speakers(following.text)
            if len(speakers) > 1 or not _same_speaker(cue.text, following.text):
                break
            joined += _words(following.text)
            last = next_end
            if joined == words and _overlap_enough(start, end, first, last):
                return True
    return False


def merge_caption_sources(original: list[Cue], recovered: list[Cue]) -> list[Cue]:
    """Retain original cues exactly; only remove temporally matching exact OCR dialogue."""
    originals = _ordered(original)
    additions = []
    for cue in _ordered(recovered):
        if _already_original(cue, originals):
            continue
        start, end = _milliseconds(cue.start), _milliseconds(cue.end)
        for index, prior in enumerate(additions):
            if (_words(prior.text) == _words(cue.text)
                    and _same_speaker(prior.text, cue.text)
                    and min(_milliseconds(prior.end), end) > max(_milliseconds(prior.start), start)):
                additions[index] = replace(prior, start=_timestamp(min(_milliseconds(prior.start), start)),
                                           end=_timestamp(max(_milliseconds(prior.end), end)))
                break
        else:
            additions.append(cue)
    return _ordered(originals + additions)


def _composition(cues: list[Cue]) -> tuple[str, str]:
    if len(cues) == 1:
        return cues[0].text, cues[0].style
    text = '\n'.join(cue.text if _TURN.match(cue.text) else '- ' + cue.text for cue in cues)
    styles = {cue.style for cue in cues}
    return text, styles.pop() if len(styles) == 1 else 'Default'


def compose_caption_timeline(atoms: list[Cue]) -> list[Cue]:
    """Sweep half-open intervals, preserving each voice until its own end boundary."""
    atoms = _ordered(atoms)
    starts, ends = {}, {}
    for index, atom in enumerate(atoms):
        starts.setdefault(_milliseconds(atom.start), []).append(index)
        ends.setdefault(_milliseconds(atom.end), []).append(index)
    boundaries = sorted(starts.keys() | ends.keys())
    active = set()
    timeline = []
    for start, end in zip(boundaries, boundaries[1:]):
        active.difference_update(ends.get(start, []))
        active.update(starts.get(start, []))
        if not active:
            continue
        text, style = _composition([atoms[index] for index in sorted(active)])
        if (timeline and timeline[-1].end == _timestamp(start)
                and (timeline[-1].text, timeline[-1].style) == (text, style)):
            timeline[-1] = replace(timeline[-1], end=_timestamp(end))
        else:
            timeline.append(Cue(_timestamp(start), _timestamp(end), text, style))
    return timeline


def _display_words(text: str) -> str:
    return ' '.join(_TURN.sub('', text).split())


def validate_caption_timeline(atoms: list[Cue], timeline: list[Cue]) -> None:
    """Raise if display wrapping changed ordered dialogue or its temporal coverage."""
    expected = compose_caption_timeline(atoms)
    previous_end = -1
    normalized = []
    for cue in timeline:
        start, end = _milliseconds(cue.start), _milliseconds(cue.end)
        if start < previous_end or end <= start or not cue.text.strip():
            raise ValueError('Caption timeline has unordered, overlapping, or empty intervals')
        previous_end = end
        normalized.append(replace(cue, text=_display_words(cue.text), style='Default'))
    expected = [replace(cue, text=_display_words(cue.text), style='Default') for cue in expected]
    if compose_caption_timeline(normalized) != compose_caption_timeline(expected):
        raise ValueError('Caption timeline changed dialogue or its timing')
