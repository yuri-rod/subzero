import importlib.util
from pathlib import Path
from types import SimpleNamespace

from subzero.timing import Report

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("refill", REPO / "scripts" / "refill.py")
refill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refill)


class Jelly:
    def __init__(self, items, media=None, broken=()):
        self._items = items
        self._media = media or {}
        self._broken = set(broken)

    def all_items(self):
        return self._items

    def media(self, item_id):
        if item_id in self._broken:
            raise RuntimeError("EBML header parsing failed")
        return self._media[item_id]


def _media(path, name="Filme"):
    return SimpleNamespace(path=path, name=name, sidecars=(), embedded=())


def test_missing_media_skips_transcode_tmp_and_broken_items():
    jelly = Jelly(
        [{"Id": "tmp", "Name": "junk"}, {"Id": "bad", "Name": "Old"}, {"Id": "ok", "Name": "Filme"}],
        media={"tmp": _media("/m/_transcoding_x.tmp"),
               "ok": _media("/m/Filme.mkv")},
        broken=("bad",),
    )
    found = refill.missing_media(jelly, "", "pt-BR")
    assert [m.name for m in found] == ["Filme"]


def test_check_sync_continues_past_a_broken_file(monkeypatch, tmp_path, capsys):
    import subzero.reference
    import subzero.sync

    good = tmp_path / "good.pt-BR.srt"
    good.write_text("1\n00:00:01,000 --> 00:00:02,000\nOi\n", encoding="utf-8")
    bad = tmp_path / "bad.pt-BR.srt"
    bad.write_text("1\n00:00:01,000 --> 00:00:02,000\nOi\n", encoding="utf-8")

    def build_reference(path, cache):
        if "bad" in str(path):
            raise RuntimeError("EBML header parsing failed")
        return {"speech": []}

    monkeypatch.setattr(subzero.reference, "build_reference", build_reference)
    monkeypatch.setattr(subzero.reference, "verify_text",
                        lambda text, ref, phase=0: Report("pass", "ok"))
    monkeypatch.setattr(subzero.sync, "auto_sync_file",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no resync needed")))

    refill.check_sync([(_media("/v/bad.mkv"), bad), (_media("/v/good.mkv"), good)])
    out = capsys.readouterr().out
    assert "bad.pt-BR.srt: verify failed" in out
    assert "good.pt-BR.srt: in sync" in out
