"""Independent local readings tied to captured caption frames."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
from difflib import SequenceMatcher
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


def _image_similarity(raw1: bytes, raw2: bytes) -> float:
    if raw1 == raw2:
        return 1.0
    if len(raw1) < 24 or len(raw2) < 24 or raw1[:8] != b"\x89PNG\r\n\x1a\n" or raw2[:8] != b"\x89PNG\r\n\x1a\n":
        return 0.0
    w1, h1 = int.from_bytes(raw1[16:20], "big"), int.from_bytes(raw1[20:24], "big")
    w2, h2 = int.from_bytes(raw2[16:20], "big"), int.from_bytes(raw2[20:24], "big")
    if (w1, h1) != (w2, h2):
        return 0.0
    idat1, idat2 = raw1.find(b"IDAT"), raw2.find(b"IDAT")
    if idat1 < 0 or idat2 < 0:
        return 0.0
    l1, l2 = int.from_bytes(raw1[idat1 - 4:idat1], "big"), int.from_bytes(raw2[idat2 - 4:idat2], "big")
    try:
        d1 = zlib.decompress(raw1[idat1 + 4:idat1 + 4 + l1])
        d2 = zlib.decompress(raw2[idat2 + 4:idat2 + 4 + l2])
    except Exception:
        return 0.0
    sub1 = memoryview(d1)[::32]
    sub2 = memoryview(d2)[::32]
    count = min(len(sub1), len(sub2))
    if count == 0:
        return 0.0
    diff = sum(abs(a - b) for a, b in zip(sub1[:count], sub2[:count]))
    return 1.0 - (diff / (count * 255.0))


def _frame_text(frame: CaptionFrame) -> str:
    if not isinstance(frame.evidence, dict):
        return ""
    native = frame.evidence.get("native_full")
    if not isinstance(native, dict):
        return ""
    text = native.get("acceptedText") or native.get("subtitleText") or ""
    return text.strip()


def _frames_similar(prev_item: tuple, curr_item: tuple) -> bool:
    prev_frame, prev_raw, _, _ = prev_item
    curr_frame, curr_raw, _, _ = curr_item
    if prev_frame.image_digest == curr_frame.image_digest:
        return True
    text1, text2 = _frame_text(prev_frame), _frame_text(curr_frame)
    if text1 and text2:
        words1 = re.findall(r"[^\W_]+", text1.lower())
        words2 = re.findall(r"[^\W_]+", text2.lower())
        if words1 and words1 == words2:
            return True
        if len(text1) >= 6 and len(text2) >= 6:
            if SequenceMatcher(None, text1.lower(), text2.lower()).ratio() >= 0.80:
                return True
    return _image_similarity(prev_raw, curr_raw) >= 0.84


def _cluster_pending_frames(pending: list[tuple]) -> list[list[tuple]]:
    if not pending:
        return []
    clusters = []
    current = [pending[0]]
    for item in pending[1:]:
        prev = current[-1]
        time_delta = item[0].timestamp - prev[0].timestamp
        total_duration = item[0].timestamp - current[0][0].timestamp
        if 0 <= time_delta <= 0.35 and total_duration <= 6.0 and _frames_similar(prev, item):
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)
    return clusters


def _is_high_confidence_passthrough(cluster: list[tuple]) -> tuple[bool, str]:
    if len(cluster) < 2:
        return False, ""
    texts = []
    for frame, _, _, _ in cluster:
        evidence = frame.evidence if isinstance(frame.evidence, dict) else {}
        if not evidence.get("admitted"):
            return False, ""
        native = evidence.get("native_full")
        if not isinstance(native, dict):
            return False, ""
        acc = native.get("acceptedText", "").strip()
        if not acc or "_" in acc or len(acc) < 3:
            return False, ""
        items = native.get("items")
        if not items or not isinstance(items, list):
            return False, ""
        for item in items:
            if not isinstance(item, dict) or float(item.get("confidence", 0)) < 0.90:
                return False, ""
        texts.append(acc)
    first_words = re.findall(r"[^\W_]+", texts[0].lower())
    if not first_words:
        return False, ""
    for text in texts[1:]:
        words = re.findall(r"[^\W_]+", text.lower())
        if words != first_words:
            return False, ""
    return True, texts[0]


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
    def __init__(self, model: str, url: str, cache_dir: str | Path, model_digest: str, *,
                 concurrency: int = 1, deduplicate: bool = True):
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
        self.concurrency = max(1, int(concurrency))
        self.deduplicate = bool(deduplicate)
        self._host, self._port = host, port
        self.prompt = (
            "Transcribe only the text shown in this image."
            if "moondream" in model.lower()
            else CAPTION_PROMPT
        )
        self.options = (
            {"temperature": 0, "num_ctx": 2048, "num_predict": 128}
            if "moondream" in model.lower()
            else CAPTION_OPTIONS
        )
        self.identity = hashlib.sha256(_encoded({
            "version": 2, "model": model, "digest": model_digest, "prompt": self.prompt,
            "crop": CAPTION_CROP, "options": self.options, "think": False, "keep_alive": "2m",
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

    def _text(self, body: dict) -> str:
        text = body.get("response")
        if (body.get("done") is not True or body.get("done_reason") != "stop"
                or not isinstance(text, str) or len(text.encode("utf-8")) > 2048
                or any(ord(character) < 32 and character not in "\n\r\t" for character in text)):
            raise RuntimeError("Local caption recognition returned an incomplete or invalid caption")
        text = text.strip()
        if "moondream" in self.model.lower():
            match = re.search(r'["“]([^"”\n]+)["”]', text)
            if match:
                return match.group(1).strip()
            if "no text" in text.lower() or "no visible" in text.lower():
                return ""
        return text

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

        clusters = _cluster_pending_frames(pending) if self.deduplicate else [[item] for item in pending]

        pending_llm_clusters = []
        for cluster in clusters:
            can_pass, pass_text = _is_high_confidence_passthrough(cluster)
            if can_pass:
                for frame, raw, cache, windows in cluster:
                    proof = {
                        "response": pass_text,
                        "model": "apple-vision-passthrough",
                        "done": True,
                        "done_reason": "stop"
                    }
                    self._save_evidence(cache, frame, raw, proof)
                    cache.write(windows, [(frame.timestamp, pass_text)], fps=1)
                    readings[frame.timestamp] = pass_text
                if progress:
                    progress(len(readings), len(frames))
            else:
                pending_llm_clusters.append(cluster)

        if not pending_llm_clusters:
            if progress:
                progress(len(frames), len(frames))
            return readings

        def execute_cluster(cluster):
            key_idx = len(cluster) // 2
            key_frame, key_raw, key_cache, key_windows = cluster[key_idx]
            body = self._request("/api/generate", {
                "model": self.model, "prompt": self.prompt,
                "images": [base64.b64encode(key_raw).decode("ascii")],
                "stream": False, "think": False, "keep_alive": "2m", "options": self.options,
            })
            self._save_evidence(key_cache, key_frame, key_raw, body)
            for frame, raw, cache, windows in cluster:
                if frame != key_frame:
                    self._save_evidence(cache, frame, raw, body)
            text = self._text(body)
            return cluster, text, body

        workers = min(self.concurrency, len(pending_llm_clusters))
        completed_clusters = []
        with compute_phase("ollama", ollama_url=self.url):
            self._verify_model()
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(execute_cluster, c) for c in pending_llm_clusters]
                    for future in as_completed(futures):
                        cluster, text, body = future.result()
                        completed_clusters.append((cluster, text, body))
                        if progress:
                            done_count = len(readings) + sum(len(c) for c, _, _ in completed_clusters)
                            progress(done_count, len(frames))
            else:
                for cluster in pending_llm_clusters:
                    cluster, text, body = execute_cluster(cluster)
                    completed_clusters.append((cluster, text, body))
                    if progress:
                        done_count = len(readings) + sum(len(c) for c, _, _ in completed_clusters)
                        progress(done_count, len(frames))
            self._verify_model()

        for cluster, text, body in completed_clusters:
            for frame, raw, cache, windows in cluster:
                cache.write(windows, [(frame.timestamp, text)], fps=1)
                readings[frame.timestamp] = text

        return readings


