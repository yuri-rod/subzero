from subzero.worker.jellyfin import JellyfinClient, parse_fps


def test_parse_fps_reads_fractions_and_plain_values():
    assert parse_fps("24000/1001") == 24000 / 1001
    assert parse_fps("25") == 25.0
    assert parse_fps(23.976) == 23.976
    assert parse_fps(None) is None
    assert parse_fps("nonsense") is None
    assert parse_fps("24/0") is None


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes

    def request(self, method, url, **kwargs):
        status, payload = self.routes[(method, url)]
        return FakeResponse(status, payload)


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_media_carries_the_video_frame_rate():
    base = "http://jf:8096"
    item = "abc"
    http = FakeHTTP({
        ("GET", f"{base}/Items/{item}/PlaybackInfo"): (200, {"MediaSources": [{
            "Path": "/v/m.mkv", "Name": "m", "Container": "mkv",
            "RunTimeTicks": 600_000_000, "Id": "src1",
            "MediaStreams": [
                {"Type": "Video", "AverageFrameRate": "24000/1001"},
                {"Type": "Audio", "Language": "eng"},
            ]}]}),
        ("GET", f"{base}/Items"): (200, {"Items": [
            {"Name": "M", "Type": "Movie", "ProviderIds": {"Imdb": "tt123"}}]}),
    })
    media = JellyfinClient(base, "key", http=http).media(item)

    assert media.fps == 24000 / 1001
    assert media.imdb_id == "tt123"
