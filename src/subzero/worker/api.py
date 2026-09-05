import hmac
import os
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import __version__
from .config import Config
from .jellyfin import JellyfinClient, JellyfinError
from .jobs import KINDS, Job, JobStore, Runner
from .moviehash import moviehash
from .notify import Notifier
from .opensubs import OpenSubtitles, OpenSubtitlesError
from .service import Service
from .srt import dump
from .tracks import ModelHolder, Ollama, transcribe
from .watch import Watcher, excluded, has_language


class JobRequest(BaseModel):
    itemId: str
    kind: str
    targetLang: str
    sourceId: str | None = None


class SearchRequest(BaseModel):
    itemId: str
    query: str | None = None
    langs: list[str] = []


def free_vram_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return int(out.stdout.strip().splitlines()[0])


async def read_capped(upload: UploadFile, cfg: Config) -> bytes:
    """Le o upload com teto: sem isso um POST grande enche o disco do windows-pc."""
    cap = cfg.asr_max_mb * 1024 * 1024
    data = b""
    while chunk := await upload.read(1024 * 1024):
        data += chunk
        if len(data) > cap:
            raise HTTPException(status_code=413, detail=f"audio acima de {cfg.asr_max_mb} MB")
    if not data:
        raise HTTPException(status_code=400, detail="audio_file vazio")
    return data


def transcribe_upload(data: bytes, holder: ModelHolder):
    """Grava o upload num temporario porque o faster-whisper le de caminho."""
    tmp = tempfile.NamedTemporaryFile(prefix="asr-", suffix=".bin", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return transcribe(tmp.name, holder, lambda p, n: None)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def job_json(job: Job) -> dict:
    return {"id": job.id, "itemId": job.item_id, "kind": job.kind, "targetLang": job.target_lang,
            "sourceId": job.source_id, "origin": job.origin,
            "state": "failed" if job.state == "needs_review" else job.state,
            "outcome": job.state, "phase": job.phase,
            "percent": job.percent, "message": job.message, "resultPath": job.result_path,
            "attempts": job.attempts, "created": job.created, "updated": job.updated}


def create_app(cfg: Config, runner: bool = True, jellyfin=None, opensubs=None, watcher=None,
              notifier=None) -> FastAPI:
    app = FastAPI(title="srt-worker", version=__version__)
    jellyfin = jellyfin or JellyfinClient(cfg.jellyfin_url, cfg.jellyfin_key,
                                          bare_lang=cfg.bare_lang)
    opensubs = opensubs or OpenSubtitles(cfg.opensubtitles_key,
                                         username=cfg.opensubtitles_user,
                                         password=cfg.opensubtitles_password)
    notifier = notifier or Notifier(cfg.ntfy_url, cfg.ntfy_topic)
    store = JobStore(cfg.db_path)
    store.requeue_running()
    # sem login a cota e a basica da chave; um erro aqui nunca impede o worker de subir
    try:
        opensubs.login()
    except (OpenSubtitlesError, AttributeError):
        pass
    service = Service(jellyfin=jellyfin, opensubs=opensubs,
                      holder=ModelHolder(cfg.whisper_model, cfg.whisper_device,
                                         cfg.whisper_compute_type or None),
                      ollama=Ollama(cfg.ollama_url, cfg.ollama_model),
                      bare_lang=cfg.bare_lang)
    from .syncflow import SyncFlow
    service.sync_flow = SyncFlow(store,service,cfg)

    app.state.cfg = cfg
    app.state.store = store
    app.state.service = service
    app.state.stop = threading.Event()
    app.state.last_activity = time.time()

    @app.middleware("http")
    async def track_activity(request: Request, call_next):
        app.state.last_activity = time.time()
        return await call_next(request)

    def require_token(request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token, cfg.bearer_token):
            raise HTTPException(status_code=401, detail="token invalido")

    guard = [Depends(require_token)]

    @app.get("/health", dependencies=guard)
    def health() -> dict:
        thread = getattr(app.state, "runner_thread", None)
        return {"version": __version__, "gpu": free_vram_mb(), "model": cfg.whisper_model,
                "auto": cfg.auto_enabled, "queued": len(store.active()),
                "runner": bool(thread and thread.is_alive())}

    @app.get("/media/{item_id}", dependencies=guard)
    def media(item_id: str) -> dict:
        found = resolve(item_id)
        return {"itemId": found.item_id, "name": found.name, "container": found.container,
                "duration": found.duration, "audioLang": found.audio_lang,
                "embedded": [{"index": s.index, "lang": s.lang, "codec": s.codec,
                              "title": s.title, "external": s.external} for s in found.embedded],
                "sidecars": found.sidecars}

    @app.post("/search", dependencies=guard)
    def search(req: SearchRequest) -> dict:
        found = resolve(req.itemId)
        try:
            digest = moviehash(found.path)
        except (OSError, ValueError):
            digest = None
        try:
            candidates = opensubs.search(query=req.query or found.name, moviehash=digest,
                                         langs=req.langs or cfg.auto_langs)
        except OpenSubtitlesError as err:
            raise HTTPException(status_code=502, detail=str(err)) from err
        return {"candidates": [{"fileId": c.file_id, "release": c.release, "lang": c.lang,
                                "downloads": c.downloads, "hearingImpaired": c.hearing_impaired,
                                "fromTrusted": c.from_trusted, "hashMatch": c.hash_match}
                               for c in candidates]}

    @app.post("/jobs", dependencies=guard)
    def create_job(req: JobRequest) -> dict:
        if req.kind not in KINDS:
            raise HTTPException(status_code=400, detail=f"tipo invalido: {req.kind}")
        found = resolve(req.itemId)
        if excluded(found.path, cfg.excluded_paths):
            raise HTTPException(status_code=403, detail="essa biblioteca esta fora do escopo")
        return job_json(store.enqueue(req.itemId, req.kind, req.targetLang, req.sourceId))

    @app.get("/jobs", dependencies=guard)
    def list_jobs(limit: int = 50) -> dict:
        return {"jobs": [job_json(j) for j in store.recent(limit)],
                "downloadsToday": store.downloads_today(),
                "budget": cfg.daily_download_budget}

    @app.get('/sync/audits', dependencies=guard)
    def sync_audits() -> dict:
        with store._db() as db:
            rows = db.execute('SELECT video,lang,status,report,updated FROM subtitle_audits'
                              ' ORDER BY updated DESC LIMIT 100').fetchall()
        return {'auditOnly':cfg.sync_audit_only,'audits':[dict(r) for r in rows]}

    @app.get("/jobs/{job_id}", dependencies=guard)
    def read_job(job_id: str) -> dict:
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job desconhecido")
        return job_json(job)

    @app.delete("/jobs/{job_id}", dependencies=guard)
    def cancel_job(job_id: str) -> dict:
        job = store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job desconhecido")
        store.cancel(job_id)
        return job_json(store.get(job_id))

    @app.get("/coverage", dependencies=guard)
    def coverage(lang: str = "pt-BR") -> dict:
        items = jellyfin.all_items()
        missing = []
        for it in items:
            try:
                media = jellyfin.media(it["Id"])
            except JellyfinError:
                continue
            if excluded(media.path, cfg.excluded_paths):
                continue
            if not has_language(media, lang):
                missing.append({"itemId": it["Id"], "name": it.get("Name") or media.name})
        return {"lang": lang, "total": len(items), "missing": missing}

    @app.post("/sweep", dependencies=guard)
    def sweep() -> dict:
        w = watcher
        if w is None:
            w = Watcher(jellyfin, store, opensubs, cfg.state_path, cfg.auto_langs,
                        cfg.daily_download_budget, fallback_langs=cfg.fallback_langs,
                        translate_from=cfg.translate_from,
                        excluded_paths=cfg.excluded_paths,sync_flow=service.sync_flow)
        enqueued = w.sweep()
        return {"enqueued": len(enqueued), "jobs": [job_json(j) for j in enqueued]}

    # Contrato do whisper-asr-webservice, que e o que o Bazarr sabe falar. Fica
    # atras do mesmo bearer do resto: o Bazarr nao manda header, entao quem quiser
    # usar poe a injecao do Authorization no proxy e nao expoe isso aberto. Sem o
    # token, qualquer um na internet enfileira GPU de graca.
    if cfg.asr_compat:
        @app.post("/asr", dependencies=guard)
        async def asr(audio_file: UploadFile = File(...), output: str = "srt",
                      task: str = "transcribe", language: str | None = None) -> Response:
            cues, _ = await run_in_threadpool(transcribe_upload, await read_capped(audio_file, cfg),
                                              service.holder)
            body = dump(cues) if output != "txt" else "\n".join(c.text for c in cues)
            return Response(content=body, media_type="text/plain; charset=utf-8")

        @app.post("/detect-language", dependencies=guard)
        async def detect_language(audio_file: UploadFile = File(...)) -> dict:
            _, language = await run_in_threadpool(transcribe_upload,
                                                  await read_capped(audio_file, cfg), service.holder)
            return {"detected_language": language, "language_code": language}

    def resolve(item_id: str):
        try:
            return jellyfin.media(item_id)
        except JellyfinError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err

    @app.post("/shutdown", dependencies=guard)
    def shutdown() -> dict:
        def _terminate():
            time.sleep(0.5)
            server = getattr(app.state, "server", None)
            if server is not None:
                server.should_exit = True
            elif not os.environ.get("PYTEST_CURRENT_TEST"):
                os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=_terminate, name="shutdown-trigger", daemon=True).start()
        return {"status": "shutting_down"}

    if runner:
        if watcher is None and cfg.auto_enabled:
            watcher = Watcher(jellyfin, store, opensubs, cfg.state_path, cfg.auto_langs,
                              cfg.daily_download_budget, fallback_langs=cfg.fallback_langs,
                              translate_from=cfg.translate_from,
                              excluded_paths=cfg.excluded_paths,sync_flow=service.sync_flow)
        start_runner(app, store, service, watcher, cfg, jellyfin, notifier)
    return app


def start_runner(app: FastAPI, store: JobStore, service: Service, watcher, cfg: Config,
                 jellyfin, notifier: Notifier) -> None:
    runner = Runner(store, {kind: service.run for kind in KINDS})

    def in_window(hour: int) -> bool:
        start_h = cfg.auto_window_start
        end_h = cfg.auto_window_end
        if start_h <= end_h:
            return start_h <= hour < end_h
        return hour >= start_h or hour < end_h

    def loop() -> None:
        last_watch = 0.0
        last_sweep_day = None
        last_busy = time.time()
        idle_limit = cfg.idle_shutdown_minutes * 60 if cfg.idle_shutdown_minutes > 0 else 0
        while not app.state.stop.is_set():
            # nada aqui pode escapar: se esta thread morre, a fila inteira congela e o
            # /health continua respondendo 200, entao ninguem fica sabendo
            now_dt = datetime.now()
            current_hour = now_dt.hour
            today_str = now_dt.strftime("%Y-%m-%d")

            allow_auto = cfg.auto_enabled and in_window(current_hour)

            if allow_auto and watcher and last_sweep_day != today_str and current_hour == cfg.auto_window_start:
                last_sweep_day = today_str
                try:
                    watcher.sweep()
                except Exception:
                    pass

            try:
                job = runner.run_once(allow_auto=allow_auto)
            except Exception:
                app.state.stop.wait(5)
                continue
            if job:
                last_busy = time.time()
                if job.state in ("done", "failed", "needs_review"):
                    name = None
                    try:
                        name = jellyfin.media(job.item_id).name
                    except Exception:
                        pass
                    try:
                        notifier.job_finished(job, item_name=name)
                    except Exception:
                        pass
            now = time.time()
            if watcher and allow_auto and not job and now - last_watch > cfg.watch_interval:
                last_watch = now
                try:
                    watcher.tick()
                except Exception:  # o laco automatico nunca derruba a fila manual
                    pass
            if idle_limit > 0 and not job:
                if len(store.active()) == 0:
                    latest_act = max(last_busy, getattr(app.state, "last_activity", 0.0))
                    outside_win = not in_window(current_hour)
                    sweep_done = in_window(current_hour) and (last_sweep_day == today_str or not watcher)
                    if (outside_win or sweep_done) and (now - latest_act >= idle_limit):
                        def _auto_term():
                            time.sleep(0.5)
                            server = getattr(app.state, "server", None)
                            if server is not None:
                                server.should_exit = True
                            elif not os.environ.get("PYTEST_CURRENT_TEST"):
                                os.kill(os.getpid(), signal.SIGTERM)
                        threading.Thread(target=_auto_term, name="idle-shutdown", daemon=True).start()
                        break
            if not job:
                app.state.stop.wait(2)

    thread = threading.Thread(target=loop, name="srt-runner", daemon=True)
    thread.start()
    app.state.runner_thread = thread

    @app.on_event("shutdown")
    def stop() -> None:
        app.state.stop.set()
