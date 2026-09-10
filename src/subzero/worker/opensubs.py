import base64
import gzip
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone

API = "https://api.opensubtitles.com/api/v1"
AGENT = "YUCAST v1.0"


def _parse_reset(value) -> float | None:
    """Parse the quota reset time as an epoch or ISO timestamp."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        stamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def norm_imdb(value) -> str:
    """Normalize IMDb IDs to digits without the tt prefix or leading zeros."""
    text = str(value or "").strip()
    if text[:2].lower() == "tt":
        text = text[2:]
    text = text.lstrip("0")
    return text


class OpenSubtitlesError(RuntimeError):
    pass


class QuotaExceeded(OpenSubtitlesError):
    def __init__(self, remaining: int = 0):
        super().__init__(f"cota diaria do OpenSubtitles estourada, restam {remaining}")
        self.remaining = remaining


@dataclass
class Candidate:
    file_id: int
    release: str
    lang: str
    downloads: int
    hearing_impaired: bool
    from_trusted: bool
    hash_match: bool
    # Match title identity using metadata rather than release-name parsing.
    ai_translated: bool = False
    machine_translated: bool = False
    imdb_id: str = ""
    tmdb_id: str = ""
    season: int | None = None
    episode: int | None = None
    feature_type: str = ""
    # Forced tracks contain partial dialogue and rank below complete subtitles.
    forced: bool = False

    @property
    def human(self) -> bool:
        return not (self.ai_translated or self.machine_translated)


@dataclass
class Account:
    level: str
    allowed: int
    used: int
    remaining: int

    @property
    def vip(self) -> bool:
        return "vip" in (self.level or "").lower()


class OpenSubtitles:
    """OpenSubtitles API v1 client.

    Credentials select the account quota and the host returned by login.
    Without credentials, requests use the API key quota."""

    # Space requests by 0.3 seconds to stay below 40 requests per 10 seconds.
    MIN_INTERVAL = 0.3

    def __init__(self, api_key: str, http=None, username: str = "", password: str = "",
                 sleep=time.sleep, clock=time.monotonic):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.token: str | None = None
        self.base = API
        self.remaining: int | None = None
        # Keep exhausted quota paused until its reset time or a fresh quota response.
        self.reset_at: float | None = None
        self._sleep = sleep
        self._clock = clock
        self._last_call = 0.0
        if http is None:
            import httpx
            # Follow redirects when the API normalizes query casing.
            http = httpx.Client(timeout=60, follow_redirects=True)
        self.http = http

    @property
    def headers(self) -> dict:
        base = {"Api-Key": self.api_key, "User-Agent": AGENT, "Accept": "application/json"}
        if self.token:
            base["Authorization"] = f"Bearer {self.token}"
        return base

    @staticmethod
    def _json(r):
        """Convert non-JSON responses into OpenSubtitlesError for callers.

        Maintenance and redirect pages can return HTML with a successful status."""
        try:
            return r.json()
        except ValueError:
            status = getattr(r, "status_code", "?")
            body = (getattr(r, "text", "") or "")[:160].replace("\n", " ")
            raise OpenSubtitlesError(
                f"resposta nao-JSON do OpenSubtitles ({status}): {body}") from None

    @staticmethod
    def clean(params: dict) -> list[tuple[str, str]]:
        """Sort query parameters, lowercase values, and drop empty entries.

        Canonical URLs avoid case-normalization redirects and share CDN cache entries."""
        return sorted((k, str(v).lower()) for k, v in params.items()
                      if v is not None and str(v) != "")

    def _wait_turn(self) -> None:
        gap = self._clock() - self._last_call
        if gap < self.MIN_INTERVAL:
            self._sleep(self.MIN_INTERVAL - gap)
        self._last_call = self._clock()

    def _call(self, method: str, url: str, tries: int = 3, **kwargs):
        headers = {**self.headers, **kwargs.pop("headers", {})}
        for attempt in range(tries):
            self._wait_turn()
            r = self.http.request(method, url, headers=headers, **kwargs)
            # Retry temporary rate limits; daily quota exhaustion needs a longer pause.
            if r.status_code == 429 and attempt < tries - 1:
                delay = r.headers.get("retry-after") if hasattr(r, "headers") else None
                try:
                    self._sleep(float(delay))
                except (TypeError, ValueError):
                    self._sleep(1.0 * (attempt + 1))
                continue
            break
        if r.status_code in (406, 429):
            body = self._json(r) if callable(getattr(r, "json", None)) else {}
            raise QuotaExceeded(int((body or {}).get("remaining", 0)))
        if r.status_code >= 400:
            raise OpenSubtitlesError(f"{method} {url} -> {r.status_code}: {r.text[:200]}")
        return r

    def login(self) -> Account | None:
        """Log in when credentials are configured; otherwise use the API key quota."""
        if not (self.username and self.password):
            return None
        payload = self._json(self._call(
            "POST", f"{API}/login",
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"}))
        token = payload.get("token")
        if not token:
            raise OpenSubtitlesError("login sem token na resposta")
        self.token = token
        user = payload.get("user") or {}
        # Use the account-specific host returned at login.
        host = (payload.get("base_url") or "").strip()
        if host:
            self.base = f"https://{host}/api/v1" if "://" not in host else f"{host.rstrip('/')}/api/v1"
        allowed = int(user.get("allowed_downloads") or 0)
        return Account(level=str(user.get("level") or ""), allowed=allowed,
                       used=0, remaining=allowed)

    def logout(self) -> bool:
        """Log out without raising, and clear the local token even if the request fails."""
        try:
            if not self.token:
                return False
            r = self.http.request("DELETE", f"{self.base}/logout", headers=self.headers)
            ok = r.status_code < 400
        except Exception:                                 # noqa: BLE001
            ok = False
        finally:
            self.token = None
        return ok

    def account(self) -> Account:
        payload = self._json(self._call("GET", f"{self.base}/infos/user")).get("data") or {}
        allowed = int(payload.get("allowed_downloads") or 0)
        used = int(payload.get("downloads_count") or 0)
        remaining = int(payload.get("remaining_downloads") or max(0, allowed - used))
        self.remaining = remaining
        return Account(level=str(payload.get("level") or ""), allowed=allowed,
                       used=used, remaining=remaining)

    def quota_exhausted(self) -> bool:
        """Check the cached quota without making another request.

        Automatic downloads stay paused until the known reset time or a quota update."""
        if self.remaining is None or self.remaining > 0:
            return False
        if self.reset_at is None:
            return True
        return self.reset_at > self._clock()

    def search(self, query: str | None = None, moviehash: str | None = None,
               langs: list[str] | None = None, imdb_id: str | None = None,
               tmdb_id: str | None = None, parent_imdb_id: str | None = None,
               parent_tmdb_id: str | None = None, season: int | None = None,
               episode: int | None = None, kind: str | None = None,
               filename: str | None = None, hash_only: bool = False) -> list[Candidate]:
        """Search by title IDs and episode numbers when available.

        Jellyfin names can contain only an episode number or a release suffix."""
        params: dict[str, str] = {}
        if query:
            params["query"] = query
        if moviehash:
            params["moviehash"] = moviehash
            # Include the filename with the hash to improve subtitle matching.
            if filename and not query:
                params["query"] = filename
        if langs:
            params["languages"] = ",".join(l.lower() for l in langs)
        if imdb_id:
            params["imdb_id"] = norm_imdb(imdb_id)
        if tmdb_id:
            params["tmdb_id"] = str(tmdb_id)
        if parent_imdb_id:
            params["parent_imdb_id"] = norm_imdb(parent_imdb_id)
        if parent_tmdb_id:
            params["parent_tmdb_id"] = str(parent_tmdb_id)
        if season is not None:
            params["season_number"] = str(season)
        if episode is not None:
            params["episode_number"] = str(episode)
        if kind:
            params["type"] = kind
        if hash_only and moviehash:
            # Strict matching accepts only an exact file hash.
            params["moviehash_match"] = "only"

        payload = self._json(self._call("GET", f"{self.base}/subtitles",
                                        params=self.clean(params)))
        found: list[Candidate] = []
        for row in payload.get("data", []):
            attrs = row.get("attributes", {})
            files = attrs.get("files") or []
            if not files:
                continue
            feature = attrs.get("feature_details") or {}
            found.append(Candidate(
                file_id=files[0]["file_id"],
                release=attrs.get("release", ""),
                lang=attrs.get("language", ""),
                # Prefer the current download counter over the legacy field.
                downloads=int(attrs.get("new_download_count")
                              or attrs.get("download_count") or 0),
                hearing_impaired=bool(attrs.get("hearing_impaired")),
                from_trusted=bool(attrs.get("from_trusted")),
                hash_match=bool(attrs.get("moviehash_match")),
                ai_translated=bool(attrs.get("ai_translated")),
                machine_translated=bool(attrs.get("machine_translated")),
                imdb_id=str(feature.get("imdb_id") or ""),
                tmdb_id=str(feature.get("tmdb_id") or ""),
                season=feature.get("season_number"),
                episode=feature.get("episode_number"),
                feature_type=str(feature.get("feature_type") or ""),
                forced=bool(attrs.get("foreign_parts_only")),
            ))
        # Prefer human translations and tracks that need less SDH cleanup.
        found.sort(key=lambda c: (c.hash_match, not c.hearing_impaired, c.human,
                                  c.from_trusted, c.downloads), reverse=True)
        return found

    def upload(self, text: str, lang: str, filename: str, imdb_id: str = "",
               movie_path: str = "", movie_hash: str = "", movie_bytes: int = 0) -> dict:
        """Upload a subtitle with metadata in the query and content in the body.

        Putting subtitle content in the query can exceed URL limits.
        A 409 response means the server already has the subtitle."""
        raw = text.encode("utf-8")
        params = {"sublanguageid": lang, "subhash": hashlib.md5(raw).hexdigest(),
                  "subfilename": filename}
        if imdb_id:
            params["imdbid"] = norm_imdb(imdb_id)
        if movie_path:
            # Handle Windows source paths even when the worker runs on macOS.
            params["moviefilename"] = movie_path.replace("\\", "/").rsplit("/", 1)[-1]
        if movie_hash:
            params["moviehash"] = movie_hash
        if movie_bytes:
            params["moviebytesize"] = str(movie_bytes)
        body = {"subcontent": base64.b64encode(gzip.compress(raw)).decode()}
        r = self.http.request("POST", f"{self.base}/subtitles/upload", headers=self.headers,
                              params=params, data=body)
        if r.status_code == 409:
            payload = self._json(r)
            return {"status": "duplicate", "subtitle_id": payload.get("duplicate_of")}
        if r.status_code >= 400:
            raise OpenSubtitlesError(f"upload -> {r.status_code}: {r.text[:200]}")
        payload = self._json(r)
        return {"status": payload.get("status") or "created",
                "subtitle_id": payload.get("subtitle_id"),
                "url": payload.get("download_url")}

    def download(self, file_id: int) -> str:
        r = self._call("POST", f"{self.base}/download",
                       json={"file_id": file_id, "sub_format": "srt"},
                       headers={"Content-Type": "application/json"})
        payload = self._json(r)
        self.remaining = payload.get("remaining")
        reset = _parse_reset(payload.get("reset_time_utc") or payload.get("reset_time"))
        if reset is not None:
            self.reset_at = reset
        link = payload.get("link")
        if not link:
            raise OpenSubtitlesError("resposta de download sem link")
        return self._call("GET", link).text
