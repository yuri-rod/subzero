"""Persist native caption observations independently of timing interpretation."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path

MAX_CACHE_BYTES = 5 * 1024 * 1024
MAX_TEXT_BYTES = 16 * 1024


def _encode(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')


def _number(value) -> float:
    if type(value) not in (int, float):
        raise ValueError('Caption scan timestamps and frame rate must be finite numbers')
    try:
        value = float(value)
    except OverflowError as err:
        raise ValueError('Caption scan timestamp or frame rate exceeds the numeric range') from err
    if not math.isfinite(value):
        raise ValueError('Caption scan timestamps and frame rate must be finite numbers')
    return value


def _windows(windows, fps, retry_all):
    fps = _number(fps)
    if fps <= 0 or type(retry_all) is not bool:
        raise ValueError('Caption scan frame rate must be positive and retry_all must be boolean')
    if not isinstance(windows, (list, tuple)):
        raise ValueError('Caption scan windows must be a sequence of time pairs')
    normalized = []
    previous_end = 0.0
    for window in windows:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError('Caption scan windows must contain exactly two timestamps')
        start, end = map(_number, window)
        if start < previous_end or end <= start:
            raise ValueError('Caption scan windows must be ordered, nonoverlapping, and nonnegative')
        normalized.append((start, end))
        previous_end = end
    return normalized, fps


def _detections(detections, windows):
    if not isinstance(detections, (list, tuple)):
        raise ValueError('Caption scan observations must be a sequence')
    checked = []
    previous = -1.0
    window_index = 0
    for detection in detections:
        if not isinstance(detection, (list, tuple)) or len(detection) != 2:
            raise ValueError('Caption scan observations must contain a timestamp and text')
        timestamp = _number(detection[0])
        text = detection[1]
        if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_TEXT_BYTES:
            raise ValueError('Caption scan text must be a UTF-8 string no larger than 16 KB')
        while window_index < len(windows) and timestamp >= windows[window_index][1]:
            window_index += 1
        if (timestamp <= previous or window_index == len(windows)
                or timestamp < windows[window_index][0]):
            raise ValueError('Caption scan timestamps must increase within the requested windows')
        checked.append((timestamp, text))
        previous = timestamp
    return checked


class CaptionScanCache:
    def __init__(self, root: str | Path, video_key: str, recognition_key: str):
        for key in (video_key, recognition_key):
            if not isinstance(key, str) or not key.strip() or len(key.encode('utf-8')) > 16384:
                raise ValueError('Caption scan identities must be nonempty strings no larger than 16 KB')
        self.root = Path(root).expanduser()
        self.video = self.root / hashlib.sha256(video_key.encode('utf-8')).hexdigest()
        self.folder = self.video / hashlib.sha256(recognition_key.encode('utf-8')).hexdigest()
        self.identity = (video_key, recognition_key)

    def _request(self, windows, fps, retry_all):
        windows, fps = _windows(windows, fps, retry_all)
        key = hashlib.sha256(_encode({'version': 1, 'identity': self.identity,
                                    'windows': windows, 'fps': fps, 'retry_all': retry_all})).hexdigest()
        return self.folder / f'{key}.json', key, windows

    def _directory(self, create=False):
        for directory in (self.root, self.video, self.folder):
            if create:
                directory.mkdir(mode=0o700, parents=directory == self.root, exist_ok=True)
            try:
                mode = directory.lstat().st_mode
            except FileNotFoundError:
                return False
            if not stat.S_ISDIR(mode):
                if create:
                    raise RuntimeError('Caption scan cache directory is not a regular directory or is a symlink')
                return False
        return True

    def read(self, windows, fps, retry_all=False) -> list[tuple[float, str]] | None:
        path, key, windows = self._request(windows, fps, retry_all)
        try:
            if not self._directory():
                return None
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_CACHE_BYTES:
                return None
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
            with os.fdopen(os.open(path, flags), 'rb') as handle:
                opened = os.fstat(handle.fileno())
                if not os.path.samestat(before, opened) or opened.st_size > MAX_CACHE_BYTES:
                    return None
                raw = handle.read(MAX_CACHE_BYTES + 1)
                after = os.fstat(handle.fileno())
                if (len(raw) > MAX_CACHE_BYTES or opened.st_size != after.st_size
                        or opened.st_mtime_ns != after.st_mtime_ns):
                    return None
        except FileNotFoundError:
            return None
        except OSError as err:
            if err.errno == errno.ELOOP:
                return None
            raise RuntimeError(f'Cannot read caption scan cache {path}: {err}') from err
        try:
            stored = json.loads(raw.decode('utf-8'))
            if (not isinstance(stored, dict) or set(stored) != {'version', 'key', 'detections', 'digest'}
                    or type(stored['version']) is not int or stored['version'] != 1 or stored['key'] != key):
                return None
            detections = _detections(stored['detections'], windows)
            content = {'version': 1, 'key': key, 'detections': detections}
            if stored['digest'] != hashlib.sha256(_encode(content)).hexdigest():
                return None
        except (ValueError, RecursionError):
            return None
        return detections

    def write(self, windows, detections, fps, retry_all=False) -> None:
        path, key, windows = self._request(windows, fps, retry_all)
        content = {'version': 1, 'key': key, 'detections': _detections(detections, windows)}
        stored = {**content, 'digest': hashlib.sha256(_encode(content)).hexdigest()}
        encoded = _encode(stored)
        if len(encoded) > MAX_CACHE_BYTES:
            raise ValueError('Caption scan cache exceeds the 5 MB size limit')
        tmp = None
        try:
            self._directory(create=True)
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                mode = None
            if mode is not None and not stat.S_ISREG(mode):
                raise RuntimeError('Caption scan cache entry is not a regular file or is a symlink')
            with tempfile.NamedTemporaryFile(mode='wb', dir=self.folder,
                                             prefix='.caption-scan-', suffix='.tmp', delete=False) as handle:
                tmp = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError as err:
            raise RuntimeError(f'Cannot write caption scan cache {path}: {err}') from err
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
