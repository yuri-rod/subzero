from types import SimpleNamespace
from unittest.mock import Mock

from subzero.worker import qbt_watch
from subzero.worker.qbt_watch import QBitWatcher


class FakeSeen:
    def __init__(self, values=()):
        self.values = set(values)
        self.added = []

    def __contains__(self, info_hash):
        return info_hash in self.values

    def add(self, info_hash):
        self.values.add(info_hash)
        self.added.append(info_hash)

    def save(self):
        pass


def make_cfg(**overrides):
    cfg = SimpleNamespace(
        qbt_url="http://127.0.0.1:8585",
        qbt_api_key="key",
        qbt_poll_interval=30,
        qbt_categories=["movies", "tv shows"],
        qbt_state_path="/tmp/qbt-seen.json",
        accepted_langs=["pt-BR"],
        auto_langs=["pt-BR"],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_torrent(**overrides):
    torrent = {"hash": "a" * 40, "name": "movie", "progress": 1.0, "category": "movies",
               "save_path": "/Volumes/HD-3/FILMES",
               "content_path": "/Volumes/HD-3/FILMES/The.Movie/The.Movie.mkv"}
    torrent.update(overrides)
    return torrent


def make_watcher(cfg=None, items=None, completed=None, library_map=None, seen=None):
    cfg = cfg or make_cfg()
    jellyfin = Mock()
    jellyfin.library_map.return_value = library_map or {
        "/Volumes/HD-3/FILMES": "lib-filmes",
        "/Volumes/HD-3/SERIES": "lib-series",
    }
    jellyfin.find_items_under.return_value = items or []
    store = Mock()
    store.active.return_value = []
    store.enqueue.return_value = SimpleNamespace(id="job1")
    qbt = Mock()
    qbt.completed.return_value = completed or []
    watcher = QBitWatcher(cfg, jellyfin, store, qbt=qbt, seen=seen or FakeSeen())
    return watcher, jellyfin, store


def test_tick_enqueues_audit_for_new_completed_torrent():
    torrent = make_torrent()
    watcher, jellyfin, store = make_watcher(
        completed=[torrent],
        items=[{"Id": "item-1", "Path": torrent["content_path"]}],
    )

    jobs = watcher.tick()

    assert len(jobs) == 1
    store.enqueue.assert_called_once_with("item-1", "audit", "pt-BR", origin="auto")
    jellyfin.refresh_library.assert_called_once_with("lib-filmes")


def test_tick_skips_seen_and_wrong_category():
    watcher, jellyfin, store = make_watcher(
        completed=[make_torrent(hash="seen-hash")],
        seen=FakeSeen(["seen-hash"]),
    )

    assert watcher.tick() == []
    store.enqueue.assert_not_called()


def test_tick_skips_without_matching_library():
    torrent = make_torrent(save_path="/Volumes/HD-3/GAMES",
                           content_path="/Volumes/HD-3/GAMES/game.iso")
    watcher, jellyfin, store = make_watcher(completed=[torrent], items=[])

    assert watcher.tick() == []
    store.enqueue.assert_not_called()
    jellyfin.refresh_library.assert_not_called()


def test_tick_does_not_reenqueue_active_item():
    torrent = make_torrent()
    watcher, jellyfin, store = make_watcher(
        completed=[torrent],
        items=[{"Id": "item-1", "Path": torrent["content_path"]}],
    )
    store.active.return_value = [SimpleNamespace(item_id="item-1", state="running")]

    assert watcher.tick() == []
    store.enqueue.assert_not_called()


def test_episode_pack_enqueues_each_item():
    torrent = make_torrent(content_path="/Volumes/HD-3/SERIES/Show.S01")
    items = [
        {"Id": "e1", "Path": "/Volumes/HD-3/SERIES/Show.S01/Show.S01E01.mkv"},
        {"Id": "e2", "Path": "/Volumes/HD-3/SERIES/Show.S01/Show.S01E02.mkv"},
    ]
    watcher, jellyfin, store = make_watcher(completed=[torrent], items=items)

    jobs = watcher.tick()

    assert {call.args[0] for call in store.enqueue.call_args_list} == {"e1", "e2"}
    assert len(jobs) == 2


def test_library_for_uses_longest_matching_location():
    watcher, jellyfin, store = make_watcher(library_map={
        "/Volumes/HD-3": "all",
        "/Volumes/HD-3/SERIES": "series",
    })
    assert watcher._library_for("/Volumes/HD-3/SERIES/Show/S01E01.mkv") == "series"


def test_items_for_waits_for_jellyfin_to_index(monkeypatch):
    monkeypatch.setattr(qbt_watch, "SETTLE_SECONDS", 0.1)
    monkeypatch.setattr(qbt_watch, "SETTLE_POLL_SECONDS", 0.01)
    torrent = make_torrent()
    watcher, jellyfin, store = make_watcher(completed=[torrent])

    jellyfin.find_items_under.side_effect = [
        [],
        [{"Id": "item-1", "Path": torrent["content_path"]}],
    ]

    items = watcher._items_for(torrent)

    assert [i["Id"] for i in items] == ["item-1"]
    assert jellyfin.find_items_under.call_count == 2
