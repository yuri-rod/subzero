"""OpenSubtitles 64-bit hash calculator."""

from __future__ import annotations

import os
import struct
from pathlib import Path

CHUNK = 65536
MASK = 0xFFFFFFFFFFFFFFFF


def moviehash(path: str | Path) -> str:
    """Calculate OpenSubtitles moviehash (file size + 64k head/tail 64-bit sum)."""
    filepath = Path(path)
    size = filepath.stat().st_size
    if size < CHUNK * 2:
        raise ValueError(f"file too small for OpenSubtitles hash: {size} bytes")

    value = size
    with open(filepath, "rb") as f:
        for offset in (0, size - CHUNK):
            f.seek(offset)
            buf = f.read(CHUNK)
            for (q,) in struct.iter_unpack("<Q", buf):
                value = (value + q) & MASK

    return f"{value:016x}"
