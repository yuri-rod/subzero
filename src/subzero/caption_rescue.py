"""Independent local readings tied to captured caption frames."""

from __future__ import annotations

import base64
import ctypes
from dataclasses import dataclass
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
from typing import Callable
from urllib.parse import urlsplit
import zlib

from .caption_scan_cache import CaptionScanCache
from .compute import compute_phase


CAPTION_PROMPT = (
    "Transcribe the English burned-in subtitle in this cropped video frame exactly. "
    "Preserve every visible word and punctuation mark. Preserve censorship bars as underscores. "
    "Ignore the background. Do not infer or fill in obscured words. "
    "Return only the subtitle text, with its original line breaks. "
    "If no subtitle is visible, return an empty string."
)
CAPTION_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 256}
CAPTION_CROP = {"x": 0, "y": 0.07, "width": 1, "height": 0.19, "padding": 5, "scale": 1,
                "decoder": "ImageIO"}
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024


def _encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _image_bytes(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FRAME_BYTES:
        raise RuntimeError("Caption frame must be a regular image no larger than 16 MB")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        opened = os.fstat(handle.fileno())
        raw = handle.read(MAX_FRAME_BYTES + 1)
        after = os.fstat(handle.fileno())
    if (not os.path.samestat(before, opened) or opened.st_size != after.st_size
            or opened.st_mtime_ns != after.st_mtime_ns or len(raw) > MAX_FRAME_BYTES):
        raise RuntimeError("Caption frame changed while reading")
    return raw


def crop_caption_image(raw: bytes) -> bytes:
    if sys.platform != "darwin":
        raise RuntimeError("Caption rescue image capture requires macOS ImageIO")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    io = ctypes.CDLL("/System/Library/Frameworks/ImageIO.framework/ImageIO")
    pointer = ctypes.c_void_p
    for library, name, arguments, returns in [
        (cf, "CFDataCreate", [pointer, ctypes.c_char_p, ctypes.c_long], pointer),
        (cf, "CFDataGetLength", [pointer], ctypes.c_long),
        (cf, "CFDataGetBytePtr", [pointer], pointer),
        (cf, "CFRelease", [pointer], None),
        (io, "CGImageSourceCreateWithData", [pointer, pointer], pointer),
        (io, "CGImageSourceCreateImageAtIndex", [pointer, ctypes.c_long, pointer], pointer),
        (cg, "CGImageGetDataProvider", [pointer], pointer),
        (cg, "CGDataProviderCopyData", [pointer], pointer),
        (cg, "CGImageRelease", [pointer], None),
    ] + [(cg, name, [pointer], ctypes.c_size_t) for name in
         ("CGImageGetWidth", "CGImageGetHeight", "CGImageGetBytesPerRow",
          "CGImageGetBitsPerPixel", "CGImageGetBitsPerComponent")]:
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, returns
    content = source = image = pixels = None
    try:
        content = cf.CFDataCreate(None, raw, len(raw))
        source = io.CGImageSourceCreateWithData(content, None) if content else None
        image = io.CGImageSourceCreateImageAtIndex(source, 0, None) if source else None
        if not image:
            raise RuntimeError("ImageIO could not decode captured caption frame")
        width, height = cg.CGImageGetWidth(image), cg.CGImageGetHeight(image)
        stride, bits = cg.CGImageGetBytesPerRow(image), cg.CGImageGetBitsPerPixel(image)
        if (not width or not height or width * height > 50_000_000
                or cg.CGImageGetBitsPerComponent(image) != 8 or bits not in (24, 32)
                or stride < width * (bits // 8)):
            raise RuntimeError("Captured caption frame has an unsupported bitmap layout")
        pixels = cg.CGDataProviderCopyData(cg.CGImageGetDataProvider(image))
        if not pixels or not height * stride <= cf.CFDataGetLength(pixels) <= 512 * 1024 * 1024:
            raise RuntimeError("Captured caption frame has incomplete bitmap pixels")
        bitmap = ctypes.string_at(cf.CFDataGetBytePtr(pixels), height * stride)
        top, bottom = max(0, math.floor(height * .74) - 5), min(height, math.ceil(height * .93) + 5)
        rows = []
        channels = bits // 8
        for y in range(top, bottom):
            row = bitmap[y * stride:y * stride + width * channels]
            if channels == 4:
                rgb = bytearray(width * 3)
                for channel in range(3):
                    rgb[channel::3] = row[channel::4]
                row = bytes(rgb)
            rows.append(b"\0" + row)

        def chunk(kind, payload):
            return (struct.pack(">I", len(payload)) + kind + payload
                    + struct.pack(">I", zlib.crc32(kind + payload)))

        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, bottom - top, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))
    finally:
        if pixels:
            cf.CFRelease(pixels)
        if image:
            cg.CGImageRelease(image)
        if source:
            cf.CFRelease(source)
        if content:
            cf.CFRelease(content)


@dataclass(frozen=True)
class CaptionFrame:
    timestamp: float
    image: Path
    image_digest: str
    evidence: dict


class CaptionRescue:
    def __init__(self, model: str, url: str, cache_dir: str | Path, model_digest: str):
        try:
            endpoint = urlsplit(url)
            host = "127.0.0.1" if endpoint.hostname == "localhost" else endpoint.hostname
            local = ipaddress.ip_address(host).is_loopback
            port = endpoint.port if endpoint.port is not None else 11434
        except (TypeError, ValueError):
            local = False
        if (not local or not 1 <= port <= 65535 or endpoint.scheme != "http" or endpoint.username is not None
                or endpoint.password is not None or endpoint.path not in ("", "/")
                or endpoint.query or endpoint.fragment):
            raise ValueError("Caption rescue requires a plain loopback HTTP endpoint")
        if (not isinstance(model, str) or not model.strip() or len(model) > 200
                or any(ord(character) < 32 for character in model)):
            raise ValueError("Caption rescue requires an installed model name")
        if not isinstance(model_digest, str) or re.fullmatch(r"[0-9a-f]{64}", model_digest) is None:
            raise ValueError("Caption rescue requires the installed model SHA256 digest")
        self.model = model
        self.model_digest = model_digest
        self.url = url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self._host, self._port = host, port
        self.identity = hashlib.sha256(_encoded({
            "version": 1, "model": model, "digest": model_digest, "prompt": CAPTION_PROMPT,
            "crop": CAPTION_CROP, "options": CAPTION_OPTIONS, "think": False, "keep_alive": "2m",
        })).hexdigest()

    def _request(self, path: str, payload=None) -> dict:
        connection = http.client.HTTPConnection(self._host, self._port, timeout=120)
        try:
            connection.request("GET" if payload is None else "POST", path,
                               body=None if payload is None else _encoded(payload),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if response.status != 200 or len(raw) > MAX_RESPONSE_BYTES:
                raise RuntimeError("Local caption recognition returned an invalid HTTP response")
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise RuntimeError("Local caption recognition returned invalid JSON")
            return body
        except (OSError, http.client.HTTPException, UnicodeError, ValueError) as err:
            raise RuntimeError("Local caption recognition request failed") from err
        finally:
            connection.close()

    def _verify_model(self):
        models = self._request("/api/tags").get("models")
        if not isinstance(models, list) or not any(
            isinstance(model, dict) and model.get("name") == self.model
            and model.get("digest") == self.model_digest for model in models
        ):
            raise RuntimeError("Caption recognition model digest differs from the configured installed model")

    @staticmethod
    def _text(body: dict) -> str:
        text = body.get("response")
        if (body.get("done") is not True or body.get("done_reason") != "stop"
                or not isinstance(text, str) or len(text.encode("utf-8")) > 2048
                or any(ord(character) < 32 and character not in "\n\r\t" for character in text)):
            raise RuntimeError("Local caption recognition returned an incomplete or invalid caption")
        return text.strip()

    def _save_evidence(self, cache, frame, raw, response=None):
        cache._directory(create=True)
        proof = {"frame": frame.evidence, "timestamp": frame.timestamp, "crop_sha256": frame.image_digest,
                 "recognition": self.identity,
                 "response": {key: value for key, value in response.items() if key != "context"} if response else None}
        for name, content in (("frame.png", raw), ("evidence.json", _encoded(proof))):
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache.folder, prefix=".rescue-", delete=False) as handle:
                    tmp = Path(handle.name)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, cache.folder / name)
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)

    def read_frames(self, video_key: str, frames: list[CaptionFrame], *,
                    progress: Callable[[int, int], None] | None = None) -> dict[float, str]:
        readings, seen, pending = {}, {}, []
        if progress:
            progress(0, len(frames))
        for frame in frames:
            timestamp = frame.timestamp
            if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0
                    or timestamp + .001 <= timestamp or type(frame.evidence.get("admitted")) is not bool):
                raise RuntimeError("Caption frame has invalid timing or admission evidence")
            raw = _image_bytes(frame.image)
            if hashlib.sha256(raw).hexdigest() != frame.image_digest:
                raise RuntimeError("Caption frame changed after capture")
            signature = hashlib.sha256(_encoded({"image": frame.image_digest,
                "source": frame.evidence.get("source_sha256"), "admitted": frame.evidence["admitted"],
                "admission_version": frame.evidence.get("admission_version")})).hexdigest()
            if timestamp in seen:
                if seen[timestamp] != signature:
                    raise RuntimeError("Different caption frames share one timestamp")
                continue
            seen[timestamp] = signature
            if not frame.evidence["admitted"]:
                readings[timestamp] = ""
                continue
            cache = CaptionScanCache(self.cache_dir, video_key, self.identity + ":" + signature)
            windows = [(timestamp, timestamp + .001)]
            cached = cache.read(windows, fps=1)
            if cached is not None and len(cached) == 1:
                readings[timestamp] = cached[0][1]
                continue
            self._save_evidence(cache, frame, raw)
            pending.append((frame, raw, cache, windows))
        if not pending:
            if progress:
                progress(len(frames), len(frames))
            return readings
        completed = []
        with compute_phase("ollama", ollama_url=self.url):
            self._verify_model()
            for frame, raw, cache, windows in pending:
                body = self._request("/api/generate", {
                    "model": self.model, "prompt": CAPTION_PROMPT,
                    "images": [base64.b64encode(raw).decode("ascii")],
                    "stream": False, "think": False, "keep_alive": "2m", "options": CAPTION_OPTIONS,
                })
                self._save_evidence(cache, frame, raw, body)
                text = self._text(body)
                completed.append((frame, raw, cache, windows, text, body))
                if progress:
                    progress(len(readings) + len(completed), len(frames))
            self._verify_model()
        for frame, raw, cache, windows, text, body in completed:
            cache.write(windows, [(frame.timestamp, text)], fps=1)
            readings[frame.timestamp] = text
        return readings
