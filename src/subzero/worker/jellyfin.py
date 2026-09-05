import os
from dataclasses import dataclass, field
from pathlib import Path


class JellyfinError(RuntimeError):
    pass


@dataclass
class EmbeddedSub:
    index: int
    lang: str
    codec: str
    title: str
    external: bool = False
    # posicao entre as legendas de dentro do arquivo: o Index do Jellyfin conta os
    # sidecars externos junto, entao ele nao serve de -map para o ffmpeg
    sub_index: int = 0


@dataclass
class Media:
    item_id: str
    name: str
    path: str
    container: str
    duration: float
    audio_lang: str
    source_id: str
    embedded: list[EmbeddedSub] = field(default_factory=list)
    sidecars: list[str] = field(default_factory=list)
    # identidade do titulo: o nome sozinho nao serve para buscar legenda, o Jellyfin
    # devolve coisas como "Episodio 1" ou so o sufixo do release
    kind: str = ""
    series_name: str = ""
    season: int | None = None
    episode: int | None = None
    imdb_id: str = ""
    tmdb_id: str = ""
    parent_imdb_id: str = ""
    parent_tmdb_id: str = ""


class JellyfinClient:
    def __init__(self, base_url: str, api_key: str, http=None, media_root: str | None = None,
                 bare_lang: str = ""):
        self.base_url = base_url.rstrip("/")
        # Jellyfin 12 nao aceita mais o X-Emby-Token pelado nas rotas de API
        self.headers = {"Authorization": f'MediaBrowser Token="{api_key}"'}
        self.media_root = media_root
        # "Filme.srt" nao tem tag; sem isto o worker nao ve a propria saida e refaz tudo
        self.bare_lang = bare_lang
        if http is None:
            import httpx
            http = httpx.Client(timeout=30)
        self.http = http

    def _call(self, method: str, path: str, **kwargs):
        r = self.http.request(method, f"{self.base_url}{path}", headers=self.headers, **kwargs)
        if r.status_code >= 400:
            raise JellyfinError(f"{method} {path} -> {r.status_code}")
        return r

    def media(self, item_id: str) -> Media:
        sources = self._call("GET", f"/Items/{item_id}/PlaybackInfo").json().get("MediaSources") or []
        if not sources:
            raise JellyfinError(f"item {item_id} sem media source")
        src = sources[0]
        path = src.get("Path") or ""
        if not path:
            raise JellyfinError(f"item {item_id} sem caminho em disco")

        streams = src.get("MediaStreams") or []
        audio = next((s for s in streams if s.get("Type") == "Audio"), {})
        embedded = []
        inside = 0
        for stream in sorted((s for s in streams if s.get("Type") == "Subtitle"),
                             key=lambda s: s.get("Index", 0)):
            external = bool(stream.get("IsExternal"))
            embedded.append(EmbeddedSub(index=stream.get("Index", 0),
                                        lang=(stream.get("Language") or "und"),
                                        codec=(stream.get("Codec") or ""),
                                        title=(stream.get("Title") or ""),
                                        external=external,
                                        sub_index=-1 if external else inside))
            if not external:
                inside += 1

        details = self.details(item_id)
        # o nome do MediaSource costuma ser o sufixo do release; o do item e melhor
        name = details.get("Name") or src.get("Name") or Path(path).stem

        ticks = src.get("RunTimeTicks") or 0
        ids = details.get("ProviderIds") or {}
        parent = self.details(details.get("SeriesId")) if details.get("SeriesId") else {}
        parent_ids = parent.get("ProviderIds") or {}
        return Media(item_id=item_id, name=name, path=path,
                     container=(src.get("Container") or Path(path).suffix.lstrip(".")),
                     duration=ticks / 10_000_000, audio_lang=(audio.get("Language") or "und"),
                     source_id=src.get("Id") or item_id, embedded=embedded,
                     sidecars=self.sidecars(path),
                     kind=("episode" if details.get("Type") == "Episode"
                           else "movie" if details.get("Type") == "Movie" else ""),
                     series_name=details.get("SeriesName") or parent.get("Name") or "",
                     season=details.get("ParentIndexNumber"),
                     episode=details.get("IndexNumber"),
                     imdb_id=str(ids.get("Imdb") or ""),
                     tmdb_id=str(ids.get("Tmdb") or ""),
                     parent_imdb_id=str(parent_ids.get("Imdb") or ""),
                     parent_tmdb_id=str(parent_ids.get("Tmdb") or ""))

    def details(self, item_id: str | None) -> dict:
        """GET /Items/{id} devolve 400 nesta versao do Jellyfin; a rota de lista com
        ids= e a que responde, e e ela que traz ProviderIds e temporada/episodio."""
        if not item_id:
            return {}
        try:
            found = self._call("GET", "/Items", params={
                "ids": item_id, "recursive": "true",
                "fields": "ProviderIds,Path,ParentId"}).json().get("Items") or []
        except JellyfinError:
            return {}
        return found[0] if found else {}

    def sidecars(self, video_path: str) -> list[str]:
        folder = self.media_root or os.path.dirname(video_path)
        stem = Path(video_path).stem
        found = []
        try:
            names = os.listdir(folder)
        except OSError:
            return []
        for name in sorted(names):
            if not name.lower().endswith((".srt", ".vtt", ".ass")):
                continue
            body = Path(name).stem
            if body == stem:
                if self.bare_lang:
                    found.append(self.bare_lang)
                continue
            if not body.startswith(stem + "."):
                continue
            found.append(body[len(stem) + 1:])
        return found

    def refresh(self, item_id: str) -> None:
        # ValidationOnly so revarre a pasta atras do sidecar novo; FullRefresh poe o
        # Jellyfin para extrair todas as legendas do arquivo e brigar pelo mesmo disco
        self._call("POST", f"/Items/{item_id}/Refresh",
                   params={"metadataRefreshMode": "ValidationOnly", "imageRefreshMode": "None",
                           "replaceAllMetadata": "false", "replaceAllImages": "false"})

    def recent(self, limit: int = 50) -> list[dict]:
        params = {"sortBy": "DateCreated", "sortOrder": "Descending", "limit": limit,
                  "recursive": "true", "includeItemTypes": "Movie,Episode",
                  "fields": "DateCreated,Path"}
        return self._call("GET", "/Items", params=params).json().get("Items", [])

    def all_items(self) -> list[dict]:
        params = {"recursive": "true", "includeItemTypes": "Movie,Episode", "fields": "Path"}
        return self._call("GET", "/Items", params=params).json().get("Items", [])
