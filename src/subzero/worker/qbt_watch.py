"""Wake the worker when a qBittorrent download finishes."""

from __future__ import annotations

import os
import threading
import time

from .qbittorrent import QBitTorrentClient, QBitTorrentError, SeenStore

SETTLE_SECONDS = 60
SETTLE_POLL_SECONDS = 2


class QBitWatcher:
    """Polls qBittorrent for finished downloads and enqueues the matching items."""

    def __init__(self, cfg, jellyfin, store, qbt=None, seen=None):
        self.cfg = cfg
        self.jellyfin = jellyfin
        self.store = store
        self.qbt = qbt or QBitTorrentClient(cfg.qbt_url, cfg.qbt_api_key)
        self.seen = seen or SeenStore(cfg.qbt_state_path)
        self.categories = set(cfg.qbt_categories)
        self.target_lang = (cfg.accepted_langs or cfg.auto_langs or ["pt-BR"])[0]
        self._library_cache: dict[str, str] | None = None
        self._stop = threading.Event()
        self._thread = None

    def tick(self) -> list:
        try:
            completed = self.qbt.completed(self.categories)
        except (QBitTorrentError, OSError):
            return []
        enqueued = []
        changed = False
        for torrent in completed:
            info_hash = torrent.get("hash")
            if not info_hash or info_hash in self.seen:
                continue
            items = self._items_for(torrent)
            if not items:
                # Jellyfin ainda nao indexou; deixa sem marcar para a proxima
                # passada tentar de novo, em vez de perder o gatilho para sempre
                continue
            self.seen.add(info_hash)
            changed = True
            for item in items:
                if self._already_enqueued(item["Id"]):
                    continue
                enqueued.append(self.store.enqueue(item["Id"], "audit", self.target_lang, origin="auto"))
        if changed:
            self.seen.save()
        return enqueued

    def _items_for(self, torrent) -> list[dict]:
        root = torrent.get("content_path") or torrent.get("save_path")
        if not root:
            return []
        library = self._library_for(root)
        if library is None:
            return []
        try:
            self.jellyfin.refresh_library(library)
        except Exception:
            pass
        deadline = time.monotonic() + SETTLE_SECONDS
        while True:
            try:
                items = self.jellyfin.find_items_under(root)
            except Exception:
                items = []
            if items:
                return items
            if time.monotonic() >= deadline:
                return []
            time.sleep(SETTLE_POLL_SECONDS)

    def _library_for(self, path: str) -> str | None:
        if self._library_cache is None:
            self._library_cache = self.jellyfin.library_map()
        norm = os.path.normpath(path).lower()
        best: tuple[str, str] | None = None
        for location, library_id in self._library_cache.items():
            loc = os.path.normpath(str(location)).lower()
            if norm == loc or norm.startswith(loc + os.sep):
                if best is None or len(loc) > len(best[0]):
                    best = (loc, library_id)
        return best[1] if best else None

    def _already_enqueued(self, item_id: str) -> bool:
        return any(
            job.item_id == item_id and job.state in ("queued", "running", "paused")
            for job in self.store.active()
        )

    def run(self) -> None:
        while not self._stop.wait(self.cfg.qbt_poll_interval):
            try:
                self.tick()
            except Exception:
                continue

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("qBittorrent watcher ja iniciado")
        self._thread = threading.Thread(target=self.run, name="qbt-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
