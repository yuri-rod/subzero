"""Atomic per-block translation cache shared by the repair and translate flows.

A cache folder holds one JSON entry per source block, keyed by cue offset.
Entries are only reused when the source text, digest, timing, and count all
match; anything else re-translates. Callers invalidate a whole folder by
changing any setting that feeds its digest (provider, model, prompt, limits).
"""
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

from .srt import Cue, dump, parse


def read_gap_source(path):
    before = Path(path).lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 2_000_000:
        raise RuntimeError('Gap recovery requires a regular subtitle file under 2 MB')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not os.path.samestat(before, info) or info.st_size > 2_000_000:
            raise RuntimeError('Subtitle changed before gap recovery')
        raw = handle.read(2_000_001)
    if len(raw) > 2_000_000:
        raise RuntimeError('Subtitle grew beyond the gap recovery limit')
    return raw.decode('utf-8-sig')


def read_cached_block(folder, block, name):
    """Raw translated cues for a block cache entry, or None when the entry
    is missing, corrupt, mistimed, or was translated from different source
    text. Callers remap cue numbers and run the full validation afterwards."""
    source = dump(block)
    cache = Path(folder) / f'{name}.json'
    if not cache.exists():
        return None
    try:
        cached = json.loads(read_gap_source(cache))
        text = cached.get('text') if isinstance(cached, dict) else None
        if not (isinstance(text, str) and cached.get('source') == source
                and cached.get('digest') == hashlib.sha256(text.encode('utf-8')).hexdigest()):
            return None
        candidate = parse(text)
        if len(candidate) != len(block):
            return None
        if any((a.start, a.end) != (b.start, b.end) or not b.text.strip()
               for a, b in zip(block, candidate)):
            return None
        return candidate
    except (ValueError, UnicodeError):
        return None


def write_cached_block(folder, name, source, text, heartbeat=None):
    """Store a translated block atomically; a crash can never leave a
    half-written entry behind."""
    cache = Path(folder) / f'{name}.json'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                     suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump({'source': source, 'text': text,
                       'digest': hashlib.sha256(text.encode('utf-8')).hexdigest()},
                      handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if heartbeat is not None:
                heartbeat()
            os.replace(tmp, cache)
        finally:
            handle.close()
            tmp.unlink(missing_ok=True)


class BlockCache:
    """Per-block translations backed by a cache folder.

    `valid` checks a translated block beyond timing and count (sanitizing,
    language gates); it lives with the caller because the checks depend on
    the target language and accepted variants.
    """

    def __init__(self, folder, valid):
        self.folder = Path(folder)
        self._valid = valid

    def read(self, block, name):
        candidate = read_cached_block(self.folder, block, name)
        if candidate is None:
            return None
        if not self._valid(block, candidate):
            return None
        return [Cue(a.index, b.start, b.end, b.text) for a, b in zip(block, candidate)]

    def valid(self, source, translated):
        return self._valid(source, translated)

    def write(self, name, source, text, heartbeat=None):
        write_cached_block(self.folder, name, source, text, heartbeat=heartbeat)
