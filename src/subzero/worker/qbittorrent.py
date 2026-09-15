"""qBittorrent Web API client and completion bookkeeping for the worker."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path


class QBitTorrentError(RuntimeError):
    pass


class QBitTorrentClient:
    """Minimal qBittorrent Web API access over the X-API-Key header."""

    def __init__(self, url: str, api_key: str, http=None):
        self.url = url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        if http is None:
            import httpx
            http = httpx.Client(timeout=15)
        self.http = http

    def torrents(self) -> list[dict]:
        try:
            response = self.http.get(f"{self.url}/api/v2/torrents/info", headers=self.headers)
        except Exception as err:
            raise QBitTorrentError(f"qBittorrent unreachable: {err}") from err
        if response.status_code in (401, 403):
            raise QBitTorrentError("qBittorrent API key was rejected")
        if response.status_code >= 400:
            raise QBitTorrentError(f"qBittorrent API responded {response.status_code}")
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def completed(self, categories) -> list[dict]:
        """Torrents fully downloaded and seeding, restricted to the wanted categories."""
        wanted = set(categories)
        return [
            torrent for torrent in self.torrents()
            if torrent.get("hash")
            and float(torrent.get("progress", 0)) >= 1.0
            and (torrent.get("category") or "") in wanted
        ]


class SeenStore:
    """Persist processed torrent hashes so each completion fires once."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._seen = self._load()

    def __contains__(self, info_hash: str) -> bool:
        return info_hash in self._seen

    def add(self, info_hash: str) -> None:
        self._seen.add(info_hash)

    def _load(self) -> set[str]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        if isinstance(payload, list) and all(isinstance(h, str) for h in payload):
            return set(payload)
        return set()

    def save(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(sorted(self._seen), output, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            with suppress(OSError):
                Path(tmp_name).unlink(missing_ok=True)
            raise
