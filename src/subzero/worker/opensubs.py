import base64
import gzip
import hashlib
import time
from dataclasses import dataclass

API = "https://api.opensubtitles.com/api/v1"
AGENT = "YUCAST v1.0"


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
    # o que a legenda diz ser: serve para conferir se e mesmo deste titulo, sem
    # depender de ler o nome do release
    ai_translated: bool = False
    machine_translated: bool = False
    imdb_id: str = ""
    tmdb_id: str = ""
    season: int | None = None
    episode: int | None = None
    feature_type: str = ""

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
    """Cliente da API v1.

    Com usuario e senha faz login e passa a usar a cota da conta (VIP e mil por dia
    em vez de cinco); so com a chave de API fica na cota basica. O login tambem
    devolve um base_url proprio, que e o host que a conta deve usar dali em diante.
    """

    # a doc pede no maximo 40 pedidos por 10 segundos por IP; um a cada 0,3s fica
    # bem abaixo disso e ainda deixa a fila andar
    MIN_INTERVAL = 0.3

    def __init__(self, api_key: str, http=None, username: str = "", password: str = "",
                 sleep=time.sleep, clock=time.monotonic):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.token: str | None = None
        self.base = API
        self.remaining: int | None = None
        self._sleep = sleep
        self._clock = clock
        self._last_call = 0.0
        if http is None:
            import httpx
            # a API responde 301 para a mesma rota com a query em minusculas; sem seguir
            # o redirect o corpo que chega e o HTML do 301 e nada funciona
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
        """A API as vezes devolve 200 com HTML (manutencao, bloqueio, o corpo de um
        301). Sem isso o JSONDecodeError sobe cru e derruba a varredura inteira,
        porque quem chama so espera OpenSubtitlesError."""
        try:
            return r.json()
        except ValueError:
            status = getattr(r, "status_code", "?")
            body = (getattr(r, "text", "") or "")[:160].replace("\n", " ")
            raise OpenSubtitlesError(
                f"resposta nao-JSON do OpenSubtitles ({status}): {body}") from None

    @staticmethod
    def clean(params: dict) -> list[tuple[str, str]]:
        """Parametros ordenados, minusculos e sem vazios.

        A doc pede exatamente isso para nao levar redirect: e o 301 que a gente
        levava, com query=Survivor virando query=survivor. Ordenar tambem melhora o
        aproveitamento do cache do CDN, que e por URL.
        """
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
            # 429 e o limite por segundo e passa sozinho; 406 e a cota do dia e nao passa
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
        """Autentica se houver credenciais. Sem elas segue na cota da chave de API."""
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
        # contas VIP recebem um host proprio; usar o generico custa cota e latencia
        host = (payload.get("base_url") or "").strip()
        if host:
            self.base = f"https://{host}/api/v1" if "://" not in host else f"{host.rstrip('/')}/api/v1"
        allowed = int(user.get("allowed_downloads") or 0)
        return Account(level=str(user.get("level") or ""), allowed=allowed,
                       used=0, remaining=allowed)

    def account(self) -> Account:
        payload = self._json(self._call("GET", f"{self.base}/infos/user")).get("data") or {}
        allowed = int(payload.get("allowed_downloads") or 0)
        used = int(payload.get("downloads_count") or 0)
        remaining = int(payload.get("remaining_downloads") or max(0, allowed - used))
        self.remaining = remaining
        return Account(level=str(payload.get("level") or ""), allowed=allowed,
                       used=used, remaining=remaining)

    def search(self, query: str | None = None, moviehash: str | None = None,
               langs: list[str] | None = None, imdb_id: str | None = None,
               tmdb_id: str | None = None, parent_imdb_id: str | None = None,
               parent_tmdb_id: str | None = None, season: int | None = None,
               episode: int | None = None, kind: str | None = None,
               filename: str | None = None) -> list[Candidate]:
        """Busca. Ids e temporada/episodio identificam o titulo sem depender do nome,
        que no Jellyfin as vezes e so 'Episodio 1' ou o sufixo do release."""
        params: dict[str, str] = {}
        if query:
            params["query"] = query
        if moviehash:
            params["moviehash"] = moviehash
            # a doc recomenda mandar o nome do arquivo junto do hash: melhora o casamento
            if filename and not query:
                params["query"] = filename
        if langs:
            params["languages"] = ",".join(l.lower() for l in langs)
        if imdb_id:
            params["imdb_id"] = str(imdb_id).lstrip("t")
        if tmdb_id:
            params["tmdb_id"] = str(tmdb_id)
        if parent_imdb_id:
            params["parent_imdb_id"] = str(parent_imdb_id).lstrip("t")
        if parent_tmdb_id:
            params["parent_tmdb_id"] = str(parent_tmdb_id)
        if season is not None:
            params["season_number"] = str(season)
        if episode is not None:
            params["episode_number"] = str(episode)
        if kind:
            params["type"] = kind

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
                # download_count e o contador velho e vem zerado; o que anda e o novo
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
            ))
        # traducao de gente ganha de traducao de maquina, mesmo com menos downloads
        # a limpeza de marcacao e sempre aproximada, entao e melhor nem precisar dela
        found.sort(key=lambda c: (c.hash_match, not c.hearing_impaired, c.human,
                                  c.from_trusted, c.downloads), reverse=True)
        return found

    def upload(self, text: str, lang: str, filename: str, imdb_id: str = "",
               movie_path: str = "", movie_hash: str = "", movie_bytes: int = 0) -> dict:
        """Devolve a legenda para o acervo.

        Os metadados vao na query e o conteudo no corpo: mandar o subcontent junto da
        query da 414, a URL nao aguenta o base64 de um srt inteiro. Duplicata volta
        409 e nao e erro nosso, e o servidor dizendo que ja tem essa.
        """
        raw = text.encode("utf-8")
        params = {"sublanguageid": lang, "subhash": hashlib.md5(raw).hexdigest(),
                  "subfilename": filename}
        if imdb_id:
            params["imdbid"] = str(imdb_id).lstrip("t")
        if movie_path:
            # caminho vem do Windows e o worker roda tambem no Mac: basename nao serve
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
        link = payload.get("link")
        if not link:
            raise OpenSubtitlesError("resposta de download sem link")
        return self._call("GET", link).text
