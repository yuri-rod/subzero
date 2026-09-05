import pytest
from fastapi.testclient import TestClient

from subzero.worker.api import create_app
from subzero.worker.config import Config
from subzero.worker.jellyfin import EmbeddedSub, JellyfinError, Media
from subzero.worker.opensubs import Candidate

AUTH = {"Authorization": "Bearer segredo"}


class FakeJellyfin:
    def __init__(self, items=None):
        self.items = items if items is not None else [{"Id": "abc", "Name": "Filme"}]

    def media(self, item_id):
        if item_id != "abc":
            raise JellyfinError("nao existe")
        return Media(item_id="abc", name="Filme", path="F:\\FILMES\\Filme.mkv", container="mkv",
                     duration=5400, audio_lang="jpn", source_id="src",
                     embedded=[EmbeddedSub(index=2, lang="eng", codec="subrip", title="English")],
                     sidecars=["pt-BR"])

    def refresh(self, item_id):
        pass

    def all_items(self):
        return self.items


class FakeOpenSubs:
    def search(self, query=None, moviehash=None, langs=None, **kwargs):
        return [Candidate(file_id=7, release="Filme WEB", lang="pt-br", downloads=120,
                          hearing_impaired=False, from_trusted=True, hash_match=True)]


@pytest.fixture
def client(tmp_path):
    cfg = Config.load({"JELLYFIN_URL": "http://jf", "JELLYFIN_API_KEY": "k",
                       "BEARER_TOKEN": "segredo", "DB_PATH": str(tmp_path / "jobs.db")})
    app = create_app(cfg, runner=False, jellyfin=FakeJellyfin(), opensubs=FakeOpenSubs())
    return TestClient(app)


def test_every_endpoint_needs_the_bearer(client):
    assert client.get("/media/abc").status_code == 401
    assert client.get("/jobs").status_code == 401
    assert client.post("/jobs", json={}).status_code == 401


def test_media_answers_without_leaking_the_disk_path(client):
    body = client.get("/media/abc", headers=AUTH).json()

    assert body["name"] == "Filme"
    assert body["duration"] == 5400
    assert body["audioLang"] == "jpn"
    assert body["embedded"] == [{"index": 2, "lang": "eng", "codec": "subrip",
                                 "title": "English", "external": False}]
    assert body["sidecars"] == ["pt-BR"]
    assert "path" not in body


def test_media_of_an_unknown_item_is_404(client):
    assert client.get("/media/nope", headers=AUTH).status_code == 404


def test_search_returns_candidates(client):
    body = client.post("/search", headers=AUTH, json={"itemId": "abc", "langs": ["pt-BR"]}).json()
    assert body["candidates"][0]["fileId"] == 7
    assert body["candidates"][0]["hashMatch"] is True


def test_job_round_trip(client):
    created = client.post("/jobs", headers=AUTH,
                          json={"itemId": "abc", "kind": "whisper", "targetLang": "pt-BR"}).json()

    assert created["state"] == "queued"
    listed = client.get("/jobs", headers=AUTH).json()["jobs"]
    assert [j["id"] for j in listed] == [created["id"]]
    assert client.get(f"/jobs/{created['id']}", headers=AUTH).json()["kind"] == "whisper"

    assert client.delete(f"/jobs/{created['id']}", headers=AUTH).status_code == 200
    assert client.get(f"/jobs/{created['id']}", headers=AUTH).json()["state"] == "cancelled"


def test_job_with_an_unknown_kind_is_400(client):
    r = client.post("/jobs", headers=AUTH, json={"itemId": "abc", "kind": "magica", "targetLang": "pt-BR"})
    assert r.status_code == 400


def test_job_for_an_unknown_item_is_404(client):
    r = client.post("/jobs", headers=AUTH, json={"itemId": "nope", "kind": "whisper", "targetLang": "pt-BR"})
    assert r.status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/jobs/inexistente", headers=AUTH).status_code == 404


def test_coverage_reports_items_missing_the_language(tmp_path):
    class TwoItemJellyfin(FakeJellyfin):
        def __init__(self):
            super().__init__(items=[{"Id": "abc", "Name": "Tem pt-BR"}, {"Id": "sem", "Name": "Sem legenda"}])

        def media(self, item_id):
            if item_id == "sem":
                return Media(item_id="sem", name="Sem legenda", path="F:\\FILMES\\Sem.mkv", container="mkv",
                            duration=1000, audio_lang="eng", source_id="src2", embedded=[], sidecars=[])
            return super().media(item_id)

    cfg = Config.load({"JELLYFIN_URL": "http://jf", "JELLYFIN_API_KEY": "k",
                       "BEARER_TOKEN": "segredo", "DB_PATH": str(tmp_path / "jobs.db")})
    app = create_app(cfg, runner=False, jellyfin=TwoItemJellyfin(), opensubs=FakeOpenSubs())
    body = TestClient(app).get("/coverage", headers=AUTH).json()

    assert body["total"] == 2
    assert [m["itemId"] for m in body["missing"]] == ["sem"]


def test_health_reports_whether_the_runner_is_alive(tmp_path):
    cfg = Config.load({"JELLYFIN_URL": "http://jf", "JELLYFIN_API_KEY": "k",
                       "BEARER_TOKEN": "segredo", "DB_PATH": str(tmp_path / "jobs.db")})
    client = TestClient(create_app(cfg, runner=False, jellyfin=FakeJellyfin(),
                                   opensubs=FakeOpenSubs()))

    body = client.get("/health", headers=AUTH).json()

    # sem runner a fila nao anda; o health tem de dizer isso em vez de so responder 200
    assert body["runner"] is False


def test_sweep_enqueues_jobs_via_api(tmp_path):
    class TwoItemJellyfin(FakeJellyfin):
        def __init__(self):
            super().__init__(items=[{"Id": "abc", "Name": "Tem pt-BR"}, {"Id": "sem", "Name": "Sem legenda"}])

        def media(self, item_id):
            if item_id == "sem":
                return Media(item_id="sem", name="Sem legenda", path="F:\\FILMES\\Sem.mkv", container="mkv",
                             duration=1000, audio_lang="eng", source_id="src2", embedded=[], sidecars=[])
            return super().media(item_id)

    cfg = Config.load({"JELLYFIN_URL": "http://jf", "JELLYFIN_API_KEY": "k",
                       "BEARER_TOKEN": "segredo", "DB_PATH": str(tmp_path / "jobs.db")})
    app = create_app(cfg, runner=False, jellyfin=TwoItemJellyfin(), opensubs=FakeOpenSubs())
    res = TestClient(app).post("/sweep", headers=AUTH)
    assert res.status_code == 200
    assert res.json()["enqueued"] == 2
def test_review_outcome_remains_terminal_for_existing_clients(tmp_path):
    from subzero.worker.api import job_json
    from subzero.worker.jobs import JobStore
    store=JobStore(str(tmp_path/'jobs.db'))
    job=store.enqueue('movie','rebuild','pt-BR')
    store.start(job.id);store.needs_review(job.id,'No verified subtitle')
    response=job_json(store.get(job.id))
    assert response['state']=='failed'
    assert response['outcome']=='needs_review'
