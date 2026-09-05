import json
import pathlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .jobs import Job, JobStore
from .moviehash import moviehash
from .opensubs import OpenSubtitlesError
from .service import three_letter

UI_LANG = {"por": "pt-BR", "eng": "en", "spa": "es", "jpn": "ja"}

Step = tuple[str, str, str | None]

EPISODE = re.compile(r"s(\d{1,2})[\s._-]*e(\d{1,3})", re.I)


def episode_code(text: str) -> str | None:
    """SxxExx normalizado, para casar 'Survivor.S48E01' com 'Survivor S48E01 ...'."""
    found = EPISODE.search(text or "")
    return f"s{int(found.group(1)):02d}e{int(found.group(2)):02d}" if found else None


def ui_lang(code: str) -> str:
    code = (code or "und").lower()
    return UI_LANG.get(code, code)


def has_language(media, lang: str) -> bool:
    target = three_letter(lang)
    if any(three_letter(tag) == target for tag in media.sidecars):
        return True
    return any(three_letter(s.lang) == target for s in media.embedded)


def normalise(path: str) -> str:
    return (path or "").replace("\\", "/").rstrip("/").lower()


def excluded(path: str, prefixes) -> bool:
    """Se o arquivo mora numa pasta barrada. Vale para qualquer via de entrada: uma
    biblioteca fora do escopo nao pode virar job nem por engano nem a pedido."""
    target = normalise(path)
    return any(pref and (target == normalise(pref) or target.startswith(normalise(pref) + "/"))
               for pref in prefixes)


def embedded_steps(media, target_lang: str) -> list[Step]:
    """Extrai a trilha de dentro do arquivo e traduz. So vale como plano B: uma
    legenda escrita por gente ganha de uma traducao automatica da faixa embutida."""
    inside = (next((s for s in media.embedded if not s.external), None)
              or next(iter(media.embedded), None))
    if inside is None:
        return []
    tag = ui_lang(inside.lang)
    steps: list[Step] = []
    if not any(three_letter(existing) == three_letter(tag) for existing in media.sidecars):
        steps.append(("embedded", tag, str(inside.index)))
    steps.append(("translate", target_lang, tag))
    return steps


def title_query(media) -> dict:
    """Como pedir esse titulo ao OpenSubtitles.

    Id sempre ganha do nome: o Jellyfin devolve o titulo no idioma da biblioteca e
    para muita serie isso e so "Episodio 1", que casa com qualquer coisa.
    """
    season = getattr(media, "season", None)
    episode = getattr(media, "episode", None)
    if getattr(media, "kind", "") == "episode" and season is not None and episode is not None:
        args = {"season": season, "episode": episode, "kind": "episode"}
        if getattr(media, "parent_imdb_id", ""):
            return {**args, "parent_imdb_id": media.parent_imdb_id}
        if getattr(media, "parent_tmdb_id", ""):
            return {**args, "parent_tmdb_id": media.parent_tmdb_id}
        if getattr(media, "series_name", ""):
            return {**args, "query": media.series_name}
        return {**args, "query": media.name}
    if getattr(media, "imdb_id", ""):
        return {"imdb_id": media.imdb_id, "kind": "movie"}
    if getattr(media, "tmdb_id", ""):
        return {"tmdb_id": media.tmdb_id, "kind": "movie"}
    return {"query": media.name}


SOURCES = {
    "bluray": {"bluray", "bluray", "bdrip", "brrip", "bdremux", "remux", "bd"},
    "web": {"webdl", "webrip", "web", "amzn", "nf", "dsnp", "hmax", "atvp"},
    "hdtv": {"hdtv", "pdtv"},
    "dvd": {"dvdrip", "dvd", "dvdscr"},
}
EDITIONS = {"extended", "directors", "director", "uncut", "unrated", "theatrical",
            "remastered", "imax", "final", "ultimate"}
RESOLUTIONS = {"2160p", "1080p", "720p", "480p", "4k"}


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-z]+", (text or "").lower()) if t]


def source_of(parts: list[str]) -> str:
    joined = set(parts)
    for family, tags in SOURCES.items():
        if joined & tags:
            return family
    return ""


def group_of(text: str) -> str:
    """O grupo do release, o que vem depois do ultimo hifen. Mesmo grupo quase sempre
    quer dizer mesma sincronia."""
    tail = (text or "").lower().rsplit("-", 1)
    if len(tail) != 2:
        return ""
    return re.split(r"[^0-9a-z]+", tail[1].strip())[0] if tail[1].strip() else ""


def release_score(local: str, release: str) -> int:
    """Quanto essa legenda combina com este arquivo.

    O imdb diz que e o mesmo filme, nao que e a mesma copia: um WEB-DL e um BluRay
    dividem o id e nao dividem a sincronia, e corte estendido muda tudo de lugar.
    """
    ours, theirs = tokens(local), tokens(release)
    if not theirs:
        return 0
    score = 0
    if group_of(local) and group_of(local) == group_of(release):
        score += 8
    a, b = source_of(ours), source_of(theirs)
    if a and b:
        score += 5 if a == b else -7
    ours_res = {t for t in ours if t in RESOLUTIONS}
    theirs_res = {t for t in theirs if t in RESOLUTIONS}
    if ours_res and theirs_res:
        score += 2 if ours_res & theirs_res else -1
    ours_ed = {t for t in ours if t in EDITIONS}
    theirs_ed = {t for t in theirs if t in EDITIONS}
    if ours_ed != theirs_ed:
        score -= 7
    elif ours_ed:
        score += 3
    return score


def same_title(media, candidate) -> bool:
    """A legenda diz de que titulo ela e; compara com o item em vez de ler o release.

    O feature_details do OpenSubtitles traz imdb/tmdb e, em serie, temporada e
    episodio. Isso e a fonte boa. O codigo SxxExx no nome do release fica de reserva
    para quando a legenda nao vier com id nenhum.
    """
    if getattr(media, "kind", "") == "episode":
        want_s, want_e = getattr(media, "season", None), getattr(media, "episode", None)
        if candidate.season is not None and candidate.episode is not None:
            return (int(candidate.season), int(candidate.episode)) == (want_s, want_e)
    else:
        for ours, theirs in ((getattr(media, "imdb_id", ""), candidate.imdb_id),
                             (getattr(media, "tmdb_id", ""), candidate.tmdb_id)):
            if ours and theirs and str(ours).lstrip("t") == str(theirs).lstrip("t"):
                return True
        if getattr(media, "imdb_id", "") or getattr(media, "tmdb_id", ""):
            return False

    code = episode_code(getattr(media, "name", "")) or episode_code(getattr(media, "path", "") or "")
    return bool(code) and episode_code(candidate.release) == code


def created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    text = re.sub(r"(\.\d{6})\d+", r"\1", value.strip()).replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


class Watcher:
    def __init__(self, jellyfin, store: JobStore, opensubs, state_path: str,
                 langs: list[str], budget: int = 15, fallback_langs: list[str] | None = None,
                 translate_from: list[str] | None = None,
                 excluded_paths: list[str] | None = None, sync_flow=None):
        self.jellyfin = jellyfin
        self.store = store
        self.opensubs = opensubs
        self.state_path = Path(state_path)
        self.langs = langs
        self.budget = budget
        # ultimo recurso antes do whisper: legenda em outra variante do idioma
        self.fallback_langs = fallback_langs or []
        # idiomas que valem baixar para traduzir depois
        self.translate_from = translate_from or ["en"]
        self.excluded_paths = excluded_paths or []
        self.sync_flow = sync_flow

    def marker(self) -> datetime | None:
        if not self.state_path.exists():
            return None
        try:
            return created_at(json.loads(self.state_path.read_text(encoding="utf-8")).get("marker"))
        except (OSError, ValueError):
            return None

    def save_marker(self, value: str) -> None:
        self.state_path.write_text(json.dumps({"marker": value}), encoding="utf-8")

    def tick(self) -> list[Job]:
        marker = self.marker()
        items = self.jellyfin.recent()
        fresh = [i for i in items if created_at(i.get("DateCreated")) and
                 (marker is None or created_at(i["DateCreated"]) > marker)]
        fresh.sort(key=lambda i: created_at(i["DateCreated"]))

        enqueued: list[Job] = []
        for item in fresh:
            try:
                media = self.jellyfin.media(item["Id"])
            except Exception:
                continue
            if excluded(media.path, self.excluded_paths):
                self.save_marker(item["DateCreated"])
                continue
            for lang in self.langs:
                budget_left = max(0, self.budget - self.store.downloads_today())
                for kind, target, source in self.plan(media, lang, budget_left):
                    enqueued.append(self.store.enqueue(media.item_id, kind, target, source,
                                                       origin="auto"))
            self.save_marker(item["DateCreated"])
        return enqueued

    def sweep(self) -> list[Job]:
        """Varre toda a biblioteca procurando itens sem legenda no idioma alvo."""
        items = self.jellyfin.all_items()
        enqueued: list[Job] = []
        active_keys = {(j.item_id, j.kind, j.target_lang) for j in self.store.active()}
        for item in items:
            try:
                media = self.jellyfin.media(item["Id"])
            except Exception:
                continue
            if excluded(media.path, self.excluded_paths):
                continue
            for lang in self.langs:
                if self.sync_flow is None and has_language(media, lang):
                    continue
                budget_left = max(0, self.budget - self.store.downloads_today())
                for kind, target, source in self.plan(media, lang, budget_left):
                    if (media.item_id, kind, target) in active_keys:
                        continue
                    job = self.store.enqueue(media.item_id, kind, target, source, origin="auto")
                    active_keys.add((media.item_id, kind, target))
                    enqueued.append(job)
        return enqueued

    def plan(self, media, target: str, budget_left: int) -> list[Step]:
        """A ordem que o acervo pede, do melhor para o mais caro.

        Legenda pronta em pt-BR primeiro. Depois ingles, que ainda vale traduzir,
        porque o texto de origem foi escrito por gente. Depois pt-PT, que ja e
        portugues. So entao a faixa embutida do arquivo, e por ultimo a GPU.
        """
        if self.sync_flow is not None:
            if any(j.item_id==media.item_id and j.target_lang==target for j in self.store.active()):
                return []
            if self.sync_flow.current(media,target):
                return []
            return [('audit',target,None)]
        if has_language(media, target):
            return []

        if budget_left > 0:
            found = self.pick_candidate(media, target)
            if found:
                return [("opensubtitles", target, found)]

            for source_lang in self.translate_from:
                found = self.pick_candidate(media, source_lang)
                if found:
                    return [("opensubtitles", source_lang, found),
                            ("translate", target, source_lang)]

            for other in self.fallback_langs:
                found = self.pick_candidate(media, other)
                if found:
                    # ja e portugues: entra com o nome do alvo, sem passar pelo ollama
                    return [("opensubtitles", target, found)]

        return embedded_steps(media, target) or [("whisper", target, None)]

    def pick_candidate(self, media, target: str) -> str | None:
        """Escolhe uma legenda so quando da para confiar que e do mesmo video.

        O nome que vem do Jellyfin as vezes e so o sufixo do release
        ("1080p.WEB.h264-EDITH[EZTVx.to]"), e uma busca por texto com isso volta
        qualquer coisa: um episodio de Survivor ja casou com um filme frances de
        2007. Vale o hash do arquivo; sem hash, exige que o codigo SxxExx bata.
        """
        try:
            digest = moviehash(media.path)
        except (OSError, ValueError):
            digest = None
        try:
            candidates = self.opensubs.search(
                langs=[target], moviehash=digest,
                filename=pathlib.Path(getattr(media, "path", "") or "").name or None,
                **title_query(media))
        except OpenSubtitlesError:
            return None
        if not candidates:
            return None

        exact = next((c for c in candidates if c.hash_match), None)
        if exact:
            return str(exact.file_id)

        # o id so garante que e o mesmo titulo; quem decide a copia e o nome do
        # release, e entre os equivalentes vale o mais baixado
        trusted = [c for c in candidates if same_title(media, c)]
        if not trusted:
            return None
        local = pathlib.Path(getattr(media, "path", "") or "").stem or getattr(media, "name", "")
        trusted.sort(key=lambda c: (release_score(local, c.release), c.downloads), reverse=True)
        return str(trusted[0].file_id)
