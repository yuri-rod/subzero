import pytest

from subzero.worker.opensubs import OpenSubtitles, OpenSubtitlesError, QuotaExceeded

SEARCH = {"data": [
    {"attributes": {"language": "pt-br", "release": "Filme 1080p WEB", "download_count": 900,
                    "hearing_impaired": False, "from_trusted": True, "moviehash_match": False,
                    "files": [{"file_id": 11, "file_name": "filme.srt"}]}},
    {"attributes": {"language": "pt-br", "release": "Filme BluRay", "download_count": 10,
                    "hearing_impaired": True, "from_trusted": False, "moviehash_match": True,
                    "files": [{"file_id": 22, "file_name": "filme2.srt"}]}},
    {"attributes": {"language": "en", "release": "sem arquivo", "download_count": 5,
                    "hearing_impaired": False, "from_trusted": False, "moviehash_match": False,
                    "files": []}},
]}


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs.get("params"), kwargs.get("json")))
        status, payload = self.routes[(method, url)]
        return FakeResponse(status, payload)


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = payload if isinstance(payload, str) else str(payload)

    def json(self):
        return self._payload


def test_search_maps_candidates_and_puts_hash_matches_first():
    http = FakeHTTP({("GET", "https://api.opensubtitles.com/api/v1/subtitles"): (200, SEARCH)})
    found = OpenSubtitles("k", http=http).search(query="Filme", langs=["pt-BR"])

    assert [c.file_id for c in found] == [22, 11]
    assert found[0].hash_match is True
    assert found[0].hearing_impaired is True
    assert found[1].downloads == 900


def test_search_sends_the_language_list_and_hash():
    http = FakeHTTP({("GET", "https://api.opensubtitles.com/api/v1/subtitles"): (200, {"data": []})})
    OpenSubtitles("k", http=http).search(query="Filme", moviehash="abc", langs=["pt-BR", "en"])

    # agora vai como lista ordenada de pares, minusculo, para nao levar 301
    params = dict(http.calls[0][2])
    assert params["languages"] == "pt-br,en"
    assert params["moviehash"] == "abc"
    assert params["query"] == "filme"


def test_download_follows_the_link():
    http = FakeHTTP({
        ("POST", "https://api.opensubtitles.com/api/v1/download"):
            (200, {"link": "https://dl.opensubtitles.com/x.srt", "remaining": 14}),
        ("GET", "https://dl.opensubtitles.com/x.srt"): (200, "1\n00:00:01,000 --> 00:00:02,000\nOi\n"),
    })
    os_client = OpenSubtitles("k", http=http)

    text = os_client.download(11)

    assert text.startswith("1\n")
    assert os_client.remaining == 14


def test_download_over_quota_raises():
    http = FakeHTTP({("POST", "https://api.opensubtitles.com/api/v1/download"):
                     (406, {"message": "quota exceeded", "remaining": 0})})
    with pytest.raises(QuotaExceeded):
        OpenSubtitles("k", http=http).download(11)


def test_api_key_and_agent_travel_on_every_call():
    http = FakeHTTP({("GET", "https://api.opensubtitles.com/api/v1/subtitles"): (200, {"data": []})})
    client = OpenSubtitles("chave", http=http)
    client.search(query="x")

    assert client.headers["Api-Key"] == "chave"
    assert "YUCAST" in client.headers["User-Agent"]


class HtmlResponse:
    """200 com corpo que nao e JSON: pagina de manutencao/bloqueio."""

    status_code = 200
    text = "<html><body>temporarily unavailable</body></html>"

    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


class HtmlHttp:
    def request(self, method, url, **kwargs):
        return HtmlResponse()


def test_search_turns_a_non_json_body_into_a_domain_error():
    client = OpenSubtitles("k", http=HtmlHttp())

    with pytest.raises(OpenSubtitlesError) as err:
        client.search(query="Survivor")

    assert "nao-JSON" in str(err.value)


def test_pick_candidate_survives_a_non_json_body():
    from subzero.worker.watch import Watcher

    watcher = Watcher(None, None, OpenSubtitles("k", http=HtmlHttp()), "state.json", ["pt-BR"])

    media = type("M", (), {"name": "Survivor.S48E01", "path": "F:\\x.mkv"})()
    assert watcher.pick_candidate(media, "pt-BR") is None


def test_client_follows_redirects():
    """A API devolve 301 para a mesma rota com a query em minusculas."""
    import httpx

    client = OpenSubtitles("k")

    assert client.http.follow_redirects is True
    client.http.close()


def test_non_json_error_says_the_status():
    client = OpenSubtitles("k", http=HtmlHttp())

    with pytest.raises(OpenSubtitlesError) as err:
        client.search(query="Survivor")

    assert "200" in str(err.value)


class RecordingUpload:
    def __init__(self, status=201, payload=None):
        self.status = status
        self.payload = payload or {"subtitle_id": 42, "status": "created",
                                   "download_url": "https://x/y"}
        self.seen = {}

    def request(self, method, url, **kwargs):
        self.seen = {"method": method, "url": url, "params": kwargs.get("params"),
                     "data": kwargs.get("data")}
        holder = self

        class R:
            status_code = holder.status
            text = "{}"

            def json(self):
                return holder.payload
        return R()


def test_upload_puts_metadata_in_the_query_and_content_in_the_body():
    import base64, gzip, hashlib
    http = RecordingUpload()
    c = OpenSubtitles("k", http=http)
    text = "1\n00:00:01,000 --> 00:00:02,000\nhello\n"

    out = c.upload(text, "en", "x.en.srt", imdb_id="tt3550190",
                   movie_path="F:\\S\\x.mkv", movie_hash="abc", movie_bytes=99)

    assert out == {"status": "created", "subtitle_id": 42, "url": "https://x/y"}
    params = http.seen["params"]
    assert params["sublanguageid"] == "en"
    assert params["imdbid"] == "3550190"
    assert params["moviefilename"] == "x.mkv"
    assert params["subhash"] == hashlib.md5(text.encode()).hexdigest()
    # o conteudo nao pode ir na URL: base64 de um srt inteiro estoura e volta 414
    assert "subcontent" not in params
    body = http.seen["data"]["subcontent"]
    assert gzip.decompress(base64.b64decode(body)).decode() == text


def test_upload_reports_a_duplicate_instead_of_raising():
    http = RecordingUpload(status=409, payload={"duplicate_of": 11832218})
    c = OpenSubtitles("k", http=http)

    assert c.upload("x", "en", "x.srt") == {"status": "duplicate", "subtitle_id": 11832218}


def test_clean_sorts_lowercases_and_drops_empties():
    """A doc: parametros ordenados, minusculos e sem valor padrao, senao vem 301."""
    out = OpenSubtitles.clean({"query": "Survivor", "languages": "pt-BR", "empty": "",
                               "none": None, "season_number": 47})

    assert out == [("languages", "pt-br"), ("query", "survivor"), ("season_number", "47")]


class Throttled:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def request(self, method, url, **kwargs):
        self.calls += 1
        status = self.statuses.pop(0)
        holder = self

        class R:
            status_code = status
            text = "{}"
            headers = {"retry-after": "0"}

            def json(self):
                return {"data": [], "remaining": 3}
        return R()


def test_rate_limit_backs_off_and_retries_instead_of_giving_up():
    """429 e o limite por segundo, passa sozinho; nao pode virar 'cota estourada'."""
    http = Throttled([429, 200])
    slept = []
    c = OpenSubtitles("k", http=http, sleep=slept.append, clock=lambda: 999.0)

    c.search(query="x", langs=["pt-BR"])

    assert http.calls == 2


def test_daily_quota_still_raises():
    http = Throttled([406])
    c = OpenSubtitles("k", http=http, sleep=lambda s: None, clock=lambda: 999.0)

    with pytest.raises(QuotaExceeded):
        c.search(query="x", langs=["pt-BR"])


def test_requests_are_spaced_out():
    http = Throttled([200, 200])
    slept, now = [], [0.0]
    c = OpenSubtitles("k", http=http, sleep=slept.append, clock=lambda: now[0])

    c.search(query="a", langs=["pt-BR"])
    c.search(query="b", langs=["pt-BR"])

    assert any(s > 0 for s in slept)


def test_ranking_prefers_a_subtitle_without_hearing_impaired_marks():
    from subzero.worker.opensubs import Candidate

    def cand(fid, hi, downloads):
        return Candidate(file_id=fid, release="r", lang="pt-br", downloads=downloads,
                         hearing_impaired=hi, from_trusted=False, hash_match=False)

    rows = [cand(1, True, 5000), cand(2, False, 10)]
    rows.sort(key=lambda c: (c.hash_match, not c.hearing_impaired, c.human,
                             c.from_trusted, c.downloads), reverse=True)

    assert rows[0].file_id == 2
