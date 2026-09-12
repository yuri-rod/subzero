#!/usr/bin/env python3
"""Prepare the Hy-MT2 7B GGUF with Tencent's EOS token and single-user template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct


EXPECTED = {
    "general.architecture": "hunyuan-dense",
    "hunyuan-dense.block_count": 32,
    "hunyuan-dense.embedding_length": 4096,
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.pre": "hunyuan",
    "tokenizer.ggml.bos_token_id": 127958,
    "tokenizer.ggml.eot_token_id": 127960,
    "tokenizer.ggml.padding_token_id": 127961,
    "tokenizer.ggml.seperator_token_id": 127962,
}
EXPECTED_TOKENS = {
    3: "$", 127958: "<|startoftext|>", 127960: "<|eos|>",
    127961: "<|pad|>", 127962: "<|extra_0|>", 127967: "<|extra_5|>",
}
EOS_KEY = "tokenizer.ggml.eos_token_id"
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}
MODELFILE = '''FROM ./hy-mt2-7b.gguf
TEMPLATE """<|startoftext|>{{ .Prompt }}<|extra_0|>{{ .Response }}"""
PARAMETER stop "<|eos|>"
PARAMETER stop "<|extra_5|>"
PARAMETER temperature 0
PARAMETER top_k 20
PARAMETER top_p 0.6
PARAMETER repeat_penalty 1.05
'''


class Header:
    def __init__(self, stream):
        self.stream = stream
        self.limit = min(os.fstat(stream.fileno()).st_size, 64 * 1024 * 1024)

    def read(self, size):
        if size < 0 or self.stream.tell() + size > self.limit:
            raise ValueError("GGUF header exceeds its file size or 64 MiB limit")
        raw = self.stream.read(size)
        if len(raw) != size:
            raise ValueError("Truncated GGUF header")
        return raw

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        size = self.number("Q")
        if size > 1024 * 1024:
            raise ValueError("GGUF string exceeds 1 MiB")
        return self.read(size).decode("utf-8")

    def value(self, kind, key):
        if kind in SCALARS:
            return self.number(SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind != 9:
            raise ValueError(f"Unsupported GGUF value type {kind} for {key}")
        subtype = self.number("I")
        count = self.number("Q")
        if count > 1000000 or subtype == 9:
            raise ValueError(f"Invalid GGUF array for {key}")
        if key == "tokenizer.ggml.tokens":
            if subtype != 8 or count != 128167:
                raise ValueError("Expected the Hy-MT2 7B vocabulary")
            selected = {}
            for index in range(count):
                token = self.string()
                if index in EXPECTED_TOKENS:
                    selected[index] = token
            if selected != EXPECTED_TOKENS:
                raise ValueError("Hy-MT2 7B special token strings do not match")
            return count
        if subtype == 8:
            for _ in range(count):
                self.string()
        elif subtype in SCALARS:
            self.read(count * struct.calcsize(SCALARS[subtype]))
        else:
            raise ValueError(f"Unsupported GGUF array type {subtype} for {key}")
        return None


def inspect(stream, *, eos=3):
    stream.seek(0)
    header = Header(stream)
    if header.read(4) != b"GGUF" or header.number("I") != 3:
        raise ValueError("Expected a little-endian GGUF version 3 file")
    tensors = header.number("Q")
    count = header.number("Q")
    if tensors != 354 or not 10 <= count <= 1024:
        raise ValueError("Unexpected Hy-MT2 7B tensor or metadata count")
    metadata = {}
    eos_offset = None
    for _ in range(count):
        key = header.string()
        if key in metadata:
            raise ValueError(f"Duplicate GGUF metadata key: {key}")
        kind = header.number("I")
        if key == EOS_KEY:
            if kind != 4:
                raise ValueError("EOS metadata must be uint32")
            eos_offset = stream.tell()
        metadata[key] = header.value(kind, key)
    for key, expected in {**EXPECTED, EOS_KEY: eos, "tokenizer.ggml.tokens": 128167}.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Unexpected {key}: {metadata.get(key)!r}, expected {expected!r}")
    for _ in range(tensors):
        header.string()
        dimensions = header.number("I")
        if not 1 <= dimensions <= 4:
            raise ValueError("Invalid GGUF tensor dimensions")
        for _ in range(dimensions):
            if not 1 <= header.number("Q") <= 2**32:
                raise ValueError("Invalid GGUF tensor dimension size")
        header.number("I")
        header.number("Q")
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError("Invalid GGUF tensor alignment")
    tensor_start = (stream.tell() + alignment - 1) // alignment * alignment
    if tensor_start >= os.fstat(stream.fileno()).st_size:
        raise ValueError("GGUF tensor bytes are missing")
    return {"eos_offset": eos_offset, "tensor_data_offset": tensor_start, "tensor_count": tensors}


def prepare(source: Path, stage: Path):
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as original:
        before = os.fstat(original.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("GGUF source must be a regular file")
        receipt = inspect(original)
        stage.mkdir(mode=0o700)
        corrected_path = stage / "hy-mt2-7b.gguf"
        source_hash, corrected_hash, tensor_hash = (hashlib.sha256() for _ in range(3))
        original.seek(0)
        offset = 0
        with corrected_path.open("xb") as corrected:
            while chunk := original.read(1024 * 1024):
                source_hash.update(chunk)
                if offset + len(chunk) > receipt["tensor_data_offset"]:
                    tensor_hash.update(chunk[max(0, receipt["tensor_data_offset"] - offset):])
                start = max(receipt["eos_offset"], offset)
                end = min(receipt["eos_offset"] + 4, offset + len(chunk))
                if start < end:
                    replacement = struct.pack("<I", 127960)[start - receipt["eos_offset"]:end - receipt["eos_offset"]]
                    chunk = chunk[:start - offset] + replacement + chunk[end - offset:]
                corrected_hash.update(chunk)
                corrected.write(chunk)
                offset += len(chunk)
            corrected.flush()
            os.fsync(corrected.fileno())
        after = os.fstat(original.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Source GGUF changed while preparing the copy; discard this staging directory")
        with corrected_path.open("rb") as corrected:
            if inspect(corrected, eos=127960) != receipt:
                raise ValueError("Corrected GGUF header verification failed")
            corrected.seek(0)
            verified_hash = hashlib.sha256()
            while chunk := corrected.read(1024 * 1024):
                verified_hash.update(chunk)
        if verified_hash.digest() != corrected_hash.digest():
            raise ValueError("Corrected GGUF byte verification failed")
        receipt.update({"source": str(source.resolve()), "size": offset,
                        "source_sha256": source_hash.hexdigest(),
                        "corrected_sha256": corrected_hash.hexdigest(),
                        "tensor_sha256": tensor_hash.hexdigest(),
                        "old_eos": 3, "new_eos": 127960})
        with (stage / "Modelfile").open("x") as config:
            config.write(MODELFILE)
        with (stage / "verification.json").open("x") as report:
            json.dump(receipt, report, indent=2)
            report.write("\n")
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing Hy-MT2 7B GGUF (read only)")
    parser.add_argument("stage", type=Path, nargs="?", help="new directory for the corrected copy and Modelfile")
    args = parser.parse_args()
    try:
        if args.stage is None:
            fd = os.open(args.source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as original:
                if not stat.S_ISREG(os.fstat(original.fileno()).st_mode):
                    raise ValueError("GGUF source must be a regular file")
                receipt = inspect(original)
        else:
            receipt = prepare(args.source, args.stage)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot prepare Hy-MT2 7B: {exc}\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
