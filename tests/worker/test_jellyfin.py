import pytest

from subzero.worker.jellyfin import JellyfinClient, JellyfinError

PLAYBACK = {
    "MediaSources": [{
        "Id": "src1",
        "Path": None,
        "Container": "mkv",
        "RunTimeTicks": 36_000_000_000,
        "MediaStreams": [
            {"Type": "Video", "Index": 0, "Codec": "h264"},
            {"Type": "Audio", "Index": 1, "Codec": "eac3", "Language": "jpn"},
            {"Type": "Subtitle", "Index": 2, "Codec": "subrip", "Language": "eng",
             "Title": "English", "IsExternal": False},
            {"Type": "Subtitle", "Index": 3, "Codec": "subrip", "Language": "por",
             "Title": "Forced", "IsExternal": True},
        ],
    }]
}


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        self.last_params = kwargs.get("params")
        status, payload = self.routes.get((method, url), (404, {}))
        return FakeResponse(status, payload)


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def client(routes, root=None):
    return JellyfinClient("http://jf", "k", http=FakeHTTP(routes), media_root=root)


def test_media_maps_source_and_streams(tmp_path):
    video = tmp_path / "Filme (2020).mkv"
    video.write_bytes(b"x")
    payload = {"MediaSources": [dict(PLAYBACK["MediaSources"][0], Path=str(video))]}
    c = client({("GET", "http://jf/Items/abc/PlaybackInfo"): (200, payload),
                ("GET", "http://jf/Items/abc"): (200, {"Name": "Filme"})})

    media = c.media("abc")

    assert media.path == str(video)
    assert media.container == "mkv"
    assert media.duration == 3600
    assert media.audio_lang == "jpn"
    assert [(s.index, s.lang, s.external) for s in media.embedded] == [(2, "eng", False), (3, "por", True)]


def test_media_lists_existing_sidecars(tmp_path):
    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Filme.pt-BR.srt").write_text("1")
    (tmp_path / "Outro.en.srt").write_text("1")
    payload = {"MediaSources": [dict(PLAYBACK["MediaSources"][0], Path=str(video))]}
    c = client({("GET", "http://jf/Items/abc/PlaybackInfo"): (200, payload),
                ("GET", "http://jf/Items/abc"): (200, {"Name": "Filme"})})

    assert c.media("abc").sidecars == ["pt-BR"]


def test_media_without_a_source_fails():
    c = client({("GET", "http://jf/Items/abc/PlaybackInfo"): (200, {"MediaSources": []})})
    with pytest.raises(JellyfinError):
        c.media("abc")


def test_unauthorized_raises():
    c = client({("GET", "http://jf/Items/abc/PlaybackInfo"): (401, {})})
    with pytest.raises(JellyfinError):
        c.media("abc")


def test_refresh_only_revalidates_the_item():
    http = FakeHTTP({("POST", "http://jf/Items/abc/Refresh"): (204, {})})
    JellyfinClient("http://jf", "k", http=http).refresh("abc")
    assert ("POST", "http://jf/Items/abc/Refresh") in http.calls
    assert http.last_params["metadataRefreshMode"] == "ValidationOnly"


def test_auth_header_is_the_full_mediabrowser_form():
    c = JellyfinClient("http://jf", "chave", http=FakeHTTP({}))
    assert c.headers["Authorization"] == 'MediaBrowser Token="chave"'


def test_recent_returns_items():
    payload = {"Items": [{"Id": "1", "Name": "A", "DateCreated": "2026-08-19T00:00:00Z"}]}
    c = client({("GET", "http://jf/Items"): (200, payload)})
    assert [i["Id"] for i in c.recent()] == ["1"]


def test_all_items_ignores_date_and_limit():
    payload = {"Items": [{"Id": "1", "Name": "A"}, {"Id": "2", "Name": "B"}]}
    c = client({("GET", "http://jf/Items"): (200, payload)})
    assert [i["Id"] for i in c.all_items()] == ["1", "2"]
    assert "limit" not in c.http.last_params
    assert "sortBy" not in c.http.last_params


def test_sub_index_skips_the_external_sidecars(tmp_path):
    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    streams = [
        {"Type": "Video", "Index": 0, "Codec": "h264"},
        {"Type": "Audio", "Index": 1, "Codec": "eac3", "Language": "eng"},
        {"Type": "Subtitle", "Index": 2, "Codec": "subrip", "Language": "por", "IsExternal": True},
        {"Type": "Subtitle", "Index": 3, "Codec": "subrip", "Language": "eng", "IsExternal": False},
        {"Type": "Subtitle", "Index": 4, "Codec": "subrip", "Language": "spa", "IsExternal": False},
    ]
    payload = {"MediaSources": [{"Id": "s", "Path": str(video), "Container": "mkv",
                                 "RunTimeTicks": 0, "MediaStreams": streams}]}
    c = client({("GET", "http://jf/Items/abc/PlaybackInfo"): (200, payload),
                ("GET", "http://jf/Items/abc"): (200, {"Name": "Filme"})})

    embedded = c.media("abc").embedded

    assert [(s.index, s.sub_index) for s in embedded] == [(2, -1), (3, 0), (4, 1)]


def test_sidecars_sees_the_bare_file_as_the_configured_language(tmp_path):
    from subzero.worker.jellyfin import JellyfinClient

    video = tmp_path / "Filme.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Filme.srt").write_text("1", encoding="utf-8")
    (tmp_path / "Filme.en.srt").write_text("1", encoding="utf-8")

    plain = JellyfinClient("http://jf", "k", http=object())
    tagged = JellyfinClient("http://jf", "k", http=object(), bare_lang="pt-BR")

    # sem a configuracao o arquivo sem tag e invisivel, que era o bug
    assert plain.sidecars(str(video)) == ["en"]
    assert sorted(tagged.sidecars(str(video))) == ["en", "pt-BR"]
