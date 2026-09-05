import json

import pytest

from subzero.worker.jellyfin import EmbeddedSub, Media
from subzero.worker.jobs import JobStore
from subzero.worker.opensubs import Candidate, QuotaExceeded
from subzero.worker.watch import Watcher, embedded_steps


def media(embedded=(), sidecars=(), audio="eng"):
    return Media(item_id="abc", name="Filme", path="F:\\FILMES\\Filme.1080p.WEB-DL.mkv",
                 container="mkv", duration=120, audio_lang=audio, source_id="src",
                 embedded=list(embedded), sidecars=list(sidecars),
                 kind="movie", imdb_id="tt123456")


def sub(index, lang):
    return EmbeddedSub(index=index, lang=lang, codec="subrip", title="")


class PlanSubs:
    """Legendas disponiveis por idioma, para montar o plano sem rede."""

    def __init__(self, by_lang=None):
        self.by_lang = by_lang or {}
        self.asked = []

    def search(self, **kwargs):
        lang = (kwargs.get("langs") or [""])[0]
        self.asked.append(lang)
        return self.by_lang.get(lang, [])


def offer(file_id, **over):
    from subzero.worker.opensubs import Candidate
    base = dict(file_id=file_id, release="Filme.1080p.WEB-DL", lang="pt-br", downloads=10,
                hearing_impaired=False, from_trusted=False, hash_match=False,
                imdb_id="123456", tmdb_id="", season=None, episode=None, feature_type="")
    base.update(over)
    return Candidate(**base)


def planner(by_lang=None, **kw):
    return Watcher(None, None, PlanSubs(by_lang), "s.json", ["pt-BR"],
                   fallback_langs=["pt-PT"], **kw)


def test_nothing_to_do_when_the_sidecar_is_there():
    assert planner().plan(media(sidecars=["pt-BR"]), "pt-BR", 10) == []


def test_nothing_to_do_when_the_target_is_embedded():
    assert planner().plan(media(embedded=[sub(2, "por")]), "pt-BR", 10) == []


def test_a_ready_made_translation_beats_extracting_the_embedded_track():
    """A ordem que importa: legenda escrita por gente ganha da traducao automatica."""
    w = planner({"pt-BR": [offer(11)]})

    plan = w.plan(media(embedded=[sub(3, "eng")]), "pt-BR", 10)

    assert plan == [("opensubtitles", "pt-BR", "11")]


def test_english_is_downloaded_to_be_translated_when_there_is_no_portuguese():
    w = planner({"en": [offer(22)]})

    plan = w.plan(media(), "pt-BR", 10)

    assert plan == [("opensubtitles", "en", "22"), ("translate", "pt-BR", "en")]
    assert w.opensubs.asked[:2] == ["pt-BR", "en"]


def test_pt_pt_comes_after_english_and_needs_no_translation():
    w = planner({"pt-PT": [offer(33)]})

    plan = w.plan(media(), "pt-BR", 10)

    assert plan == [("opensubtitles", "pt-BR", "33")]
    assert w.opensubs.asked == ["pt-BR", "en", "pt-PT"]


def test_the_embedded_track_is_the_next_resort():
    w = planner({})

    plan = w.plan(media(embedded=[sub(3, "eng")]), "pt-BR", 10)

    assert plan == [("embedded", "en", "3"), ("translate", "pt-BR", "en")]


def test_whisper_is_the_last_resort():
    assert planner({}).plan(media(), "pt-BR", 10) == [("whisper", "pt-BR", None)]


def test_without_budget_it_never_asks_opensubtitles():
    w = planner({"pt-BR": [offer(11)]})

    plan = w.plan(media(embedded=[sub(3, "eng")]), "pt-BR", 0)

    assert w.opensubs.asked == []
    assert plan == [("embedded", "en", "3"), ("translate", "pt-BR", "en")]


def test_embedded_steps_skips_extraction_when_the_sidecar_already_exists():
    assert embedded_steps(media(embedded=[sub(3, "eng")], sidecars=["en"]), "pt-BR") == [
        ("translate", "pt-BR", "en")]


class FakeJellyfin:
    def __init__(self, items, media_by_id):
        self.items = items
        self.media_by_id = media_by_id

    def recent(self, limit=50):
        return self.items

    def media(self, item_id):
        return self.media_by_id[item_id]


class FakeOpenSubs:
    def __init__(self, candidates=(), raises=None):
        self.candidates = list(candidates)
        self.raises = raises
        self.searches = 0

    def search(self, query=None, moviehash=None, langs=None, **kwargs):
        self.searches += 1
        if self.raises:
            raise self.raises
        return self.candidates


@pytest.fixture
def store(tmp_path):
    return JobStore(str(tmp_path / "jobs.db"))


def watcher(store, tmp_path, items, media_map, opensubs=None):
    return Watcher(jellyfin=FakeJellyfin(items, media_map), store=store,
                   opensubs=opensubs or FakeOpenSubs(),
                   state_path=str(tmp_path / "watch.json"), langs=["pt-BR"], budget=15)


def test_tick_enqueues_for_new_items_only(store, tmp_path):
    items = [{"Id": "abc", "DateCreated": "2026-08-19T10:00:00.0000000Z"}]
    w = watcher(store, tmp_path, items, {"abc": media(embedded=[sub(3, "eng")])})

    first = w.tick()
    second = w.tick()

    assert [j.kind for j in first] == ["embedded", "translate"]
    assert second == []


def test_marker_survives_a_restart(store, tmp_path):
    items = [{"Id": "abc", "DateCreated": "2026-08-19T10:00:00.0000000Z"}]
    media_map = {"abc": media(embedded=[sub(3, "eng")])}
    watcher(store, tmp_path, items, media_map).tick()

    fresh = watcher(store, tmp_path, items, media_map)

    assert fresh.tick() == []
    assert json.loads(open(tmp_path / "watch.json").read())["marker"].startswith("2026-08-19")


def test_opensubtitles_plan_becomes_a_job_with_the_chosen_file(store, tmp_path):
    items = [{"Id": "abc", "DateCreated": "2026-08-19T10:00:00Z"}]
    candidates = [Candidate(file_id=42, release="r", lang="pt-br", downloads=10,
                            hearing_impaired=False, from_trusted=True, hash_match=True)]
    w = watcher(store, tmp_path, items, {"abc": media()}, opensubs=FakeOpenSubs(candidates))

    jobs = w.tick()

    assert [(j.kind, j.source_id, j.origin) for j in jobs] == [("opensubtitles", "42", "auto")]


def test_no_candidate_falls_back_to_whisper(store, tmp_path):
    items = [{"Id": "abc", "DateCreated": "2026-08-19T10:00:00Z"}]
    w = watcher(store, tmp_path, items, {"abc": media()}, opensubs=FakeOpenSubs([]))

    assert [j.kind for j in w.tick()] == ["whisper"]


def test_quota_error_falls_back_to_whisper(store, tmp_path):
    items = [{"Id": "abc", "DateCreated": "2026-08-19T10:00:00Z"}]
    w = watcher(store, tmp_path, items, {"abc": media()}, opensubs=FakeOpenSubs(raises=QuotaExceeded(0)))

    assert [j.kind for j in w.tick()] == ["whisper"]


def test_budget_counts_the_downloads_already_done_today(store, tmp_path):
    items = [{"Id": "abc", "DateCreated": "2026-08-19T10:00:00Z"}]
    for _ in range(15):
        job = store.enqueue("x", "opensubtitles", "pt-BR", origin="auto")
        store.start(job.id)
        store.finish(job.id, "x.srt")
    opensubs = FakeOpenSubs([Candidate(1, "r", "pt-br", 1, False, False, False)])
    w = watcher(store, tmp_path, items, {"abc": media()}, opensubs=opensubs)

    jobs = w.tick()

    assert [j.kind for j in jobs] == ["whisper"]
    assert opensubs.searches == 0


class OneCandidate:
    def __init__(self, candidates):
        self.candidates = candidates
        self.asked = {}

    def search(self, **kwargs):
        self.asked = kwargs
        return self.candidates


def candidate(file_id, release, hash_match=False):
    from subzero.worker.opensubs import Candidate
    return Candidate(file_id=file_id, release=release, lang="pt-br", downloads=1,
                     hearing_impaired=False, from_trusted=True, hash_match=hash_match)


def test_episode_code_normalises_separators():
    from subzero.worker.watch import episode_code

    assert episode_code("Survivor.S48E01.720p") == "s48e01"
    assert episode_code("Survivor S48E01 Committing") == "s48e01"
    assert episode_code("survivor.s48e1.x264") == "s48e01"
    assert episode_code("La.Mome.2007.DVDRip") is None


def test_pick_candidate_prefers_the_hash_match():
    from subzero.worker.watch import Watcher

    osubs = OneCandidate([candidate(1, "algo diferente"), candidate(2, "outro", hash_match=True)])
    w = Watcher(None, None, osubs, "s.json", ["pt-BR"])
    media = type("M", (), {"name": "Survivor.S48E01.720p", "path": "F:\\x.mkv"})()

    assert w.pick_candidate(media, "pt-BR") == "2"


def test_pick_candidate_refuses_an_unrelated_title():
    """O bug real: nome inutil casou com um filme frances de 2007."""
    from subzero.worker.watch import Watcher

    osubs = OneCandidate([candidate(4573756, "La.Mome.2007.DVDRip.XviD_ENG.ESP.PT-BR")])
    w = Watcher(None, None, osubs, "s.json", ["pt-BR"])
    media = type("M", (), {"name": "1080p.WEB.h264-EDITH[EZTVx.to]", "path": "F:\\y.mkv"})()

    assert w.pick_candidate(media, "pt-BR") is None


def test_pick_candidate_accepts_a_matching_episode_code():
    from subzero.worker.watch import Watcher

    osubs = OneCandidate([candidate(9, "Survivor S48E01 Committing to the Bit 1080p")])
    w = Watcher(None, None, osubs, "s.json", ["pt-BR"])
    media = type("M", (), {"name": "Survivor.S48E01.720p.HDTV.x264-JACKED", "path": "F:\\z.mkv"})()

    assert w.pick_candidate(media, "pt-BR") == "9"


def test_pick_candidate_falls_back_to_the_path_for_the_code():
    from subzero.worker.watch import Watcher

    osubs = OneCandidate([candidate(9, "Survivor S48E02 1080p")])
    w = Watcher(None, None, osubs, "s.json", ["pt-BR"])
    media = type("M", (), {"name": "1080p.WEB.h264-EDITH",
                           "path": "F:\\SERIES\\Survivor\\Season 48\\Survivor.S48E02.mkv"})()

    assert w.pick_candidate(media, "pt-BR") == "9"


def episode_media(**over):
    base = {"name": "Episódio 1", "path": "F:\\x.mkv", "kind": "episode", "season": 47,
            "episode": 1, "series_name": "Survivor", "parent_imdb_id": "tt0239195",
            "parent_tmdb_id": "14658", "imdb_id": "tt32436919", "tmdb_id": ""}
    base.update(over)
    return type("M", (), base)()


def test_title_query_uses_the_series_id_and_numbers_for_an_episode():
    from subzero.worker.watch import title_query

    assert title_query(episode_media()) == {
        "season": 47, "episode": 1, "kind": "episode", "parent_imdb_id": "tt0239195"}


def test_title_query_falls_back_to_tmdb_then_series_name():
    from subzero.worker.watch import title_query

    assert title_query(episode_media(parent_imdb_id=""))["parent_tmdb_id"] == "14658"
    q = title_query(episode_media(parent_imdb_id="", parent_tmdb_id=""))
    assert q["query"] == "Survivor"


def test_title_query_never_sends_the_localised_episode_title_alone():
    """'Episodio 1' casava com qualquer coisa; so pode ir junto de temporada e episodio."""
    from subzero.worker.watch import title_query

    q = title_query(episode_media(parent_imdb_id="", parent_tmdb_id="", series_name=""))

    assert q["query"] == "Episódio 1"
    assert q["season"] == 47 and q["episode"] == 1


def test_title_query_uses_the_movie_id():
    from subzero.worker.watch import title_query

    movie = type("M", (), {"name": "La Mome", "kind": "movie", "imdb_id": "tt0450188",
                           "tmdb_id": "4550", "season": None, "episode": None})()

    assert title_query(movie) == {"imdb_id": "tt0450188", "kind": "movie"}


def test_pick_candidate_searches_by_id_not_by_name():
    from subzero.worker.watch import Watcher

    osubs = OneCandidate([candidate(9, "Survivor S47E01 1080p", hash_match=True)])
    w = Watcher(None, None, osubs, "s.json", ["pt-BR"])

    assert w.pick_candidate(episode_media(), "pt-BR") == "9"
    assert osubs.asked["parent_imdb_id"] == "tt0239195"
    assert osubs.asked["season"] == 47 and osubs.asked["episode"] == 1
    assert "query" not in osubs.asked


def full_candidate(file_id, **over):
    from subzero.worker.opensubs import Candidate
    base = dict(file_id=file_id, release="", lang="pt-br", downloads=0, hearing_impaired=False,
                from_trusted=False, hash_match=False, imdb_id="", tmdb_id="",
                season=None, episode=None, feature_type="")
    base.update(over)
    return Candidate(**base)


def test_same_title_matches_an_episode_by_season_and_number():
    from subzero.worker.watch import same_title

    assert same_title(episode_media(), full_candidate(1, season=47, episode=1))
    assert not same_title(episode_media(), full_candidate(1, season=47, episode=2))


def test_same_title_matches_a_movie_by_id():
    from subzero.worker.watch import same_title

    movie = type("M", (), {"name": "Deep Water", "kind": "movie", "imdb_id": "tt29516222",
                           "tmdb_id": "1127384", "season": None, "episode": None, "path": ""})()

    assert same_title(movie, full_candidate(1, imdb_id="29516222"))
    assert same_title(movie, full_candidate(1, tmdb_id="1127384"))
    assert not same_title(movie, full_candidate(1, imdb_id="999999"))


def test_same_title_refuses_a_movie_whose_ids_do_not_match():
    """O caso La Mome: nome inutil, id diferente, nao pode passar."""
    from subzero.worker.watch import same_title

    movie = type("M", (), {"name": "1080p.WEB.h264-EDITH", "kind": "movie",
                           "imdb_id": "tt32436919", "tmdb_id": "", "season": None,
                           "episode": None, "path": ""})()

    assert not same_title(movie, full_candidate(4573756, imdb_id="450188", release="La.Mome.2007"))


def test_candidate_ranking_puts_human_above_ai():
    from subzero.worker.opensubs import Candidate
    rows = [full_candidate(1, downloads=900, ai_translated=True),
            full_candidate(2, downloads=10)]
    rows.sort(key=lambda c: (c.hash_match, c.human, c.from_trusted, c.downloads), reverse=True)

    assert rows[0].file_id == 2


def test_release_score_prefers_the_same_source():
    from subzero.worker.watch import release_score

    local = "Deep.Water.2026.1080p.WEB-DL.DDP5.1-EDITH"

    assert release_score(local, "Deep.Water.2026.1080p.WEB-DL.DDP5.1-EDITH") > \
           release_score(local, "Deep.Water.2026.BluRay.1080p.ReMuX.AVC.DTS-HD")


def test_release_score_punishes_a_different_cut():
    from subzero.worker.watch import release_score

    local = "Blade.Runner.1982.1080p.BluRay-GROUP"

    assert release_score(local, "Blade.Runner.1982.Final.Cut.1080p.BluRay-GROUP") < \
           release_score(local, "Blade.Runner.1982.1080p.BluRay-GROUP")


def test_release_score_rewards_the_same_group():
    from subzero.worker.watch import release_score

    local = "Survivor.S48E01.1080p.WEB.h264-EDITH"

    assert release_score(local, "Survivor.S48E01.1080p.WEB.h264-EDITH") > \
           release_score(local, "Survivor.S48E01.1080p.WEB.h264-SPAMNEGGS")


def test_pick_candidate_takes_the_matching_release_over_the_most_downloaded():
    """Mesmo imdb, copias diferentes: sincronia ganha de popularidade."""
    from subzero.worker.watch import Watcher

    ours = full_candidate(1, release="Deep.Water.2026.1080p.WEB-DL-EDITH",
                          downloads=5, imdb_id="29516222")
    popular = full_candidate(2, release="Deep.Water.2026.BluRay.1080p.ReMuX",
                             downloads=9000, imdb_id="29516222")
    osubs = OneCandidate([popular, ours])
    w = Watcher(None, None, osubs, "s.json", ["pt-BR"])
    movie = type("M", (), {"name": "Deep Water", "kind": "movie", "imdb_id": "tt29516222",
                           "tmdb_id": "", "season": None, "episode": None,
                           "path": "F:\\FILMES\\Deep.Water.2026.1080p.WEB-DL-EDITH.mkv"})()

    assert w.pick_candidate(movie, "pt-BR") == "1"


def test_pick_candidate_uses_downloads_when_releases_are_equivalent():
    from subzero.worker.watch import Watcher

    a = full_candidate(1, release="Deep.Water.2026.1080p.WEB-DL-EDITH", downloads=5,
                       imdb_id="29516222")
    b = full_candidate(2, release="Deep.Water.2026.1080p.WEB-DL-EDITH", downloads=800,
                       imdb_id="29516222")
    osubs = OneCandidate([a, b])
    w = Watcher(None, None, osubs, "s.json", ["pt-BR"])
    movie = type("M", (), {"name": "Deep Water", "kind": "movie", "imdb_id": "tt29516222",
                           "tmdb_id": "", "season": None, "episode": None,
                           "path": "F:\\FILMES\\Deep.Water.2026.1080p.WEB-DL-EDITH.mkv"})()

    assert w.pick_candidate(movie, "pt-BR") == "2"


class LangAwareSubs:
    """Devolve candidato so no idioma pedido."""

    def __init__(self, by_lang):
        self.by_lang = by_lang
        self.asked_langs = []

    def search(self, **kwargs):
        lang = (kwargs.get("langs") or [""])[0]
        self.asked_langs.append(lang)
        return self.by_lang.get(lang, [])


def movie_media():
    return type("M", (), {"name": "Deep Water", "kind": "movie", "imdb_id": "tt29516222",
                          "tmdb_id": "", "season": None, "episode": None, "item_id": "abc",
                          "path": "F:\\FILMES\\Deep.Water.2026.1080p.WEB-DL-EDITH.mkv"})()


class FakeStore:
    def __init__(self):
        self.calls = []

    def enqueue(self, item_id, kind, target, source, origin="manual"):
        self.calls.append((kind, target, source))
        return type("J", (), {"kind": kind, "target_lang": target, "source_id": source})()

    def downloads_today(self):
        return 0




def test_excluded_matches_a_whole_drive():
    from subzero.worker.watch import excluded

    assert excluded("E:\\0h2cx7tcwjjhv5jz80nm6_source.mp4", ["E:\\"])
    assert excluded("e:/sub/folder/x.mkv", ["E:\\"])
    assert not excluded("F:\\FILMES\\Filme.mkv", ["E:\\"])


def test_excluded_does_not_match_a_lookalike_prefix():
    from subzero.worker.watch import excluded

    assert not excluded("F:\\FILMES2\\x.mkv", ["F:\\FILMES"])
    assert excluded("F:\\FILMES\\x.mkv", ["F:\\FILMES"])


def test_excluded_is_false_without_any_rule():
    from subzero.worker.watch import excluded

    assert not excluded("E:\\x.mp4", [])


def test_tick_never_queues_an_excluded_item(tmp_path):
    class Jelly:
        def media(self, item_id):
            return media()  # caminho em F:, mas trocamos abaixo

    store = FakeStore()
    w = Watcher(Jelly(), store, PlanSubs(), str(tmp_path / "s.json"), ["pt-BR"],
                excluded_paths=["F:\\FILMES"])
    w.jellyfin.recent = lambda limit=50: [{"Id": "abc", "DateCreated": "2026-08-21T10:00:00Z"}]

    w.tick()

    assert store.calls == []


def test_sweep_queues_all_missing_items(tmp_path):
    class Jelly:
        def all_items(self):
            return [{"Id": "1"}, {"Id": "2"}]

        def media(self, item_id):
            if item_id == "1":
                return type("M", (), {"item_id": "1", "path": "/media/a.mkv", "sidecars": [], "embedded": []})()
            return type("M", (), {"item_id": "2", "path": "/media/b.mkv", "sidecars": ["pt-BR"], "embedded": []})()

    class Store:
        def __init__(self):
            self.calls = []
            self.enqueued = []

        def active(self):
            return []

        def downloads_today(self):
            return 0

        def enqueue(self, item_id, kind, target, source, origin="manual"):
            self.calls.append((item_id, kind, target, source, origin))
            return type("J", (), {"id": "j1", "item_id": item_id})()

    w = Watcher(Jelly(), Store(), None, str(tmp_path / "s.json"), ["pt-BR"])
    w.plan = lambda media, lang, budget: [("whisper", "pt-BR", None)]

    jobs = w.sweep()

    assert len(jobs) == 1
    assert w.store.calls == [("1", "whisper", "pt-BR", None, "auto")]

