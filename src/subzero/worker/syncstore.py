import json
import time


class SyncStore:
    def __init__(self, jobs):
        self.jobs = jobs
        with jobs._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS subtitle_attempts (
                    video TEXT, lang TEXT, file_id INTEGER, job_id TEXT,
                    created REAL, status TEXT DEFAULT 'reserved', digest TEXT DEFAULT '',
                    path TEXT DEFAULT '', report TEXT DEFAULT '{}',
                    PRIMARY KEY(video,lang,file_id)
                );
                CREATE TABLE IF NOT EXISTS subtitle_audits (
                    video TEXT, lang TEXT, digest TEXT, status TEXT, report TEXT,
                    updated REAL, PRIMARY KEY(video,lang)
                );
            ''')

    def reserve(self, video, lang, file_id, job_id):
        with self.jobs._db() as db:
            db.execute('BEGIN IMMEDIATE')
            cursor = db.execute('INSERT OR IGNORE INTO subtitle_attempts'
                                '(video,lang,file_id,job_id,created) VALUES (?,?,?,?,?)',
                                (video,lang,file_id,job_id,time.time()))
            return cursor.rowcount == 1

    def update(self, video, lang, file_id, *, status, digest='', path='', report=None):
        with self.jobs._db() as db:
            db.execute('UPDATE subtitle_attempts SET status=?, digest=?, path=?, report=?'
                       ' WHERE video=? AND lang=? AND file_id=?',
                       (status,digest,path,json.dumps(report or {}),video,lang,file_id))

    def attempts(self, video, lang):
        with self.jobs._db() as db:
            return [dict(r) for r in db.execute('SELECT * FROM subtitle_attempts'
                                              ' WHERE video=? AND lang=? ORDER BY created', (video,lang))]

    def audit(self, video, lang, digest, status, report):
        with self.jobs._db() as db:
            db.execute('INSERT OR REPLACE INTO subtitle_audits VALUES (?,?,?,?,?,?)',
                       (video,lang,digest,status,json.dumps(report),time.time()))

    def current(self, video, lang, digest):
        with self.jobs._db() as db:
            row = db.execute('SELECT * FROM subtitle_audits WHERE video=? AND lang=?',
                             (video,lang)).fetchone()
        return bool(row and row['digest']==digest and
                    (row['status']=='pass' or time.time()-row['updated'] < 86400))


def broken_file_ids(jobs_store, lang) -> set:
    """file_ids que ja voltaram vazios: o plugin do Jellyfin filtra esses dos
    automaticos para nao gastar cota com legenda quebrada duas vezes."""
    if jobs_store is None:
        return set()
    with jobs_store._db() as db:
        db.execute("CREATE TABLE IF NOT EXISTS subtitle_attempts (video TEXT, lang TEXT,"
                   " file_id INTEGER, job_id TEXT, created REAL, status TEXT DEFAULT 'reserved',"
                   " digest TEXT DEFAULT '', path TEXT DEFAULT '', report TEXT DEFAULT '{}',"
                   " PRIMARY KEY(video,lang,file_id))")
        return {row[0] for row in db.execute("SELECT file_id FROM subtitle_attempts"
                                             " WHERE lang=? AND status='broken'", (lang,))}
