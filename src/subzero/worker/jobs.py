import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Callable

KINDS = ("embedded", "opensubtitles", "whisper", "translate",
         "audit", "refetch", "resync", "embedded_translate", "rebuild")

# jobs automaticos (watcher) tentam de novo sozinhos; manuais falham na hora porque
# tem alguem olhando a tela esperando
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 60
RETRY_MAX_SECONDS = 1800

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    target_lang TEXT NOT NULL,
    source_id TEXT,
    origin TEXT NOT NULL DEFAULT 'manual',
    state TEXT NOT NULL DEFAULT 'queued',
    phase TEXT NOT NULL DEFAULT '',
    percent INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    result_path TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt REAL NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
"""


@dataclass
class Job:
    id: str
    item_id: str
    kind: str
    target_lang: str
    source_id: str | None
    origin: str
    state: str
    phase: str
    percent: int
    message: str
    result_path: str | None
    attempts: int
    next_attempt: float
    created: float
    updated: float


class JobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._db() as db:
            db.executescript(SCHEMA)
            self._migrate(db)

    def _migrate(self, db: sqlite3.Connection) -> None:
        cols = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
        if "attempts" not in cols:
            db.execute("ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        if "next_attempt" not in cols:
            db.execute("ALTER TABLE jobs ADD COLUMN next_attempt REAL NOT NULL DEFAULT 0")

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def enqueue(self, item_id: str, kind: str, target_lang: str,
                source_id: str | None = None, origin: str = "manual") -> Job:
        if kind not in KINDS:
            raise ValueError(f"tipo de job desconhecido: {kind}")
        now = time.time()
        job_id = uuid.uuid4().hex
        with self._db() as db:
            db.execute("INSERT INTO jobs (id, item_id, kind, target_lang, source_id, origin, created, updated)"
                       " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (job_id, item_id, kind, target_lang, source_id, origin, now, now))
        return self.get(job_id)

    def get(self, job_id: str) -> Job | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job(**dict(row)) if row else None

    def recent(self, limit: int = 50) -> list[Job]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [Job(**dict(r)) for r in rows]

    def active(self) -> list[Job]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE state IN ('queued', 'running') ORDER BY created").fetchall()
        return [Job(**dict(r)) for r in rows]

    def _next(self, db, allow_auto):
        if db.execute("SELECT 1 FROM jobs WHERE state = 'running' LIMIT 1").fetchone():
            return None
        origin_filter = "" if allow_auto else " AND origin = 'manual'"
        return db.execute(
            f"SELECT * FROM jobs WHERE state = 'queued' AND next_attempt <= ?{origin_filter}"
            " ORDER BY CASE origin WHEN 'manual' THEN 0 ELSE 1 END,"
            " MAX(0, CASE kind WHEN 'resync' THEN 1"
            " WHEN 'embedded_translate' THEN 2 WHEN 'translate' THEN 2"
            " WHEN 'rebuild' THEN 3 WHEN 'whisper' THEN 3 ELSE 0 END"
            " - CAST((? - created) / 3600 AS INTEGER)), created LIMIT 1",
            (time.time(),time.time())).fetchone()

    def next_queued(self, allow_auto: bool = True) -> Job | None:
        with self._db() as db:
            row = self._next(db,allow_auto)
        return Job(**dict(row)) if row else None

    def claim(self, allow_auto: bool = True) -> Job | None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._next(db,allow_auto)
            if row is None:
                return None
            db.execute("UPDATE jobs SET state='running', updated=? WHERE id=?",
                       (time.time(),row['id']))
        return self.get(row['id'])

    def advance(self, job_id, kind, message):
        if kind not in KINDS:
            raise ValueError('Unknown repair stage')
        with self._db() as db:
            db.execute("UPDATE jobs SET state='queued', kind=?, message=?, phase=?, percent=0,"
                       " source_id=NULL, next_attempt=0, updated=? WHERE id=? AND state='running'",
                       (kind,message,message,time.time(),job_id))

    def needs_review(self, job_id, message):
        with self._db() as db:
            db.execute("UPDATE jobs SET state='needs_review', message=?, phase=?, updated=?"
                       " WHERE id=? AND state='running'",(message,message,time.time(),job_id))

    def _set(self, job_id: str, **fields) -> None:
        fields["updated"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._db() as db:
            db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))

    def start(self, job_id: str) -> None:
        self._set(job_id, state="running", phase="iniciando", percent=0)

    def progress(self, job_id: str, phase: str, percent: int) -> None:
        self._set(job_id, phase=phase, percent=max(0, min(100, int(percent))))

    def finish(self, job_id: str, result_path: str | None) -> None:
        with self._db() as db:
            db.execute("UPDATE jobs SET state='done', phase='pronto', percent=100, result_path=?,"
                       " updated=? WHERE id=? AND state='running'",(result_path,time.time(),job_id))

    def fail(self, job_id: str, message: str, retryable: bool = False) -> None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
            if row is None or row['state'] != 'running':
                return
            if retryable and row['origin']=='auto' and row['attempts'] < MAX_RETRIES:
                delay = min(RETRY_BASE_SECONDS * (2 ** row['attempts']), RETRY_MAX_SECONDS)
                db.execute("UPDATE jobs SET state='queued', phase='', percent=0, message=?,"
                           " attempts=attempts+1, next_attempt=?, updated=? WHERE id=?",
                           (message[:500],time.time()+delay,time.time(),job_id))
            else:
                db.execute("UPDATE jobs SET state='failed', message=?, updated=? WHERE id=?",
                           (message[:500],time.time(),job_id))

    def cancel(self, job_id: str) -> None:
        with self._db() as db:
            db.execute("UPDATE jobs SET state = 'cancelled', updated = ? WHERE id = ? AND state IN ('queued', 'running')",
                       (time.time(), job_id))

    def requeue_running(self) -> None:
        """Servico caiu no meio de um job: ele volta para a fila em vez de sumir."""
        with self._db() as db:
            db.execute("UPDATE jobs SET state = 'queued', phase = '', percent = 0, updated = ?"
                       " WHERE state = 'running'", (time.time(),))

    def downloads_today(self) -> int:
        start = time.time() - (time.time() % 86400)
        with self._db() as db:
            row = db.execute("SELECT COUNT(*) AS n FROM jobs WHERE kind = 'opensubtitles'"
                             " AND state = 'done' AND updated >= ?", (start,)).fetchone()
            extra = 0
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='subtitle_attempts'").fetchone():
                extra = db.execute('SELECT COUNT(*) FROM subtitle_attempts WHERE created>=?',
                                   (start,)).fetchone()[0]
        return int(row["n"])+extra


Handler = Callable[[Job, Callable[[str, int], None]], str | None]


class Runner:
    def __init__(self, store: JobStore, handlers: dict[str, Handler]):
        self.store = store
        self.handlers = handlers

    def run_once(self, allow_auto: bool = True) -> Job | None:
        job = self.store.claim(allow_auto=allow_auto)
        if not job:
            return None

        handler = self.handlers.get(job.kind)
        if handler is None:
            self.store.fail(job.id, f"sem handler para {job.kind}")
            return self.store.get(job.id)

        def progress(phase: str, percent: int) -> None:
            self.store.progress(job.id, phase, percent)

        try:
            result = handler(job, progress)
        except Exception as err:
            self.store.fail(job.id, f"{type(err).__name__}: {err}", retryable=True)
        else:
            if self.store.get(job.id).state == "running":
                self.store.finish(job.id, result)
        return self.store.get(job.id)
