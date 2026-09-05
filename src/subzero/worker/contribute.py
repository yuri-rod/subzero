"""Upload subtitle batches to OpenSubtitles and track duplicates.

Candidates include worker translations and English sidecars.
Use a small batch to inspect the results before uploading more."""
import argparse
import base64
import gzip
import hashlib
import json
import pathlib
import re
import sqlite3
import sys
import time
from dataclasses import dataclass

from .config import Config, read_env_file
from .jellyfin import JellyfinClient
from .moviehash import moviehash
from .opensubs import OpenSubtitles, OpenSubtitlesError, QuotaExceeded

MT_NOTE = "MACHINE TRANSLATION"
WROTE = re.compile(r"wrote (.+?\.srt) in \d+s")


@dataclass
class Candidate:
    video: pathlib.Path
    sub: pathlib.Path
    lang: str
    translated: bool
    item_id: str


def translated_here(log_path) -> set:
    """Read translated subtitle names from the translator log.

    File modification times cannot establish who produced a subtitle."""
    log = pathlib.Path(log_path)
    if not log.exists():
        return set()
    return set(WROTE.findall(log.read_text(encoding="utf-8", errors="replace")))


def read_subtitle(path) -> str:
    raw = pathlib.Path(path).read_bytes()
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise OpenSubtitlesError(f"nao consegui ler {pathlib.Path(path).name} em utf-8 nem cp1252")


def candidates(items, mine, bare_lang):
    for it in items:
        video = pathlib.Path(it.get("Path") or "")
        if not video.suffix:
            continue
        bare = video.with_suffix(".srt")
        if bare.exists() and bare.name in mine:
            yield Candidate(video, bare, bare_lang, True, it["Id"])
        english = video.with_suffix(".en.srt")
        if english.exists() and english.stat().st_size > 1000:
            yield Candidate(video, english, "en", False, it["Id"])


def open_ledger(path):
    db = sqlite3.connect(path)
    db.execute("""create table if not exists uploads (
        subhash text primary key, path text, lang text, status text,
        subtitle_id integer, flags text, at text)""")
    db.commit()
    return db


def upload(client, cand: Candidate, imdb_id: str, text: str):
    """Upload a subtitle; treat HTTP 409 as an existing catalogue entry."""
    raw = text.encode("utf-8")
    params = {
        "sublanguageid": cand.lang,
        "subhash": hashlib.md5(raw).hexdigest(),
        "subfilename": cand.sub.name,
        "imdbid": str(imdb_id).lstrip("t"),
        "moviefilename": cand.video.name,
        "moviebytesize": str(cand.video.stat().st_size),
        "moviehash": moviehash(str(cand.video)),
        "releasename": cand.video.stem,
    }
    if cand.translated:
        # The endpoint does not apply machine-translation flags, so disclose
        # translation in the release name and comment.
        params["releasename"] = f"{MT_NOTE} - {cand.video.stem}"
        params["comment"] = ("Machine translation produced by a local LLM from the "
                             "release's English subtitle. Timings come from that "
                             "subtitle and were not re-synced.")
    # Keep subtitle content in the body to avoid the query URL size limit.
    body = {"subcontent": base64.b64encode(gzip.compress(raw)).decode()}
    r = client.http.post(f"{client.base}/subtitles/upload", headers=client.headers,
                         params=params, data=body, timeout=120, follow_redirects=True)
    if r.status_code == 409:
        return "duplicate", (r.json() or {}).get("duplicate_of"), ""
    if r.status_code >= 400:
        raise OpenSubtitlesError(f"upload -> {r.status_code}: {r.text[:160]}")
    payload = r.json()
    return (payload.get("status") or "created", payload.get("subtitle_id"),
            ",".join(payload.get("flags_applied") or []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="subzero.worker.contribute")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--log", required=True, help="log do tradutor, prova de origem")
    ap.add_argument("--ledger", default="contributions.db")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--lang", default="both", help="pt-BR, en ou both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load({**read_env_file(pathlib.Path(args.env))})
    jf = JellyfinClient(cfg.jellyfin_url, cfg.jellyfin_key, bare_lang=cfg.bare_lang)
    client = OpenSubtitles(cfg.opensubtitles_key, username=cfg.opensubtitles_user,
                           password=cfg.opensubtitles_password)
    client.login()

    db = open_ledger(args.ledger)
    sent_before = {row[0] for row in db.execute("select subhash from uploads")}
    mine = translated_here(args.log)

    tally = {"sent": 0, "duplicate": 0, "no_imdb": 0, "failed": 0}
    # Index the library once to avoid repeated path scans.
    for cand in candidates(jf.all_items(), mine, cfg.bare_lang or "pt-BR"):
        if tally["sent"] + tally["duplicate"] >= args.limit:
            break
        if args.lang != "both" and cand.lang != args.lang:
            continue
        try:
            text = read_subtitle(cand.sub)
        except OpenSubtitlesError as e:
            tally["failed"] += 1
            print(f"  FAIL {cand.sub.name[:50]}: {e}")
            continue
        # Deduplicate by content hash so renamed files remain recognizable.
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        if digest in sent_before:
            continue
        media = jf.media(cand.item_id)
        if not media.imdb_id:
            tally["no_imdb"] += 1
            continue
        if args.dry_run:
            print(f"  would send [{cand.lang}] {cand.sub.name[:58]}")
            tally["sent"] += 1
            continue
        try:
            status, sub_id, flags = upload(client, cand, media.imdb_id, text)
        except QuotaExceeded as e:
            print(f"  cota estourada: {e}")
            break
        except (OpenSubtitlesError, OSError) as e:
            tally["failed"] += 1
            print(f"  FAIL [{cand.lang}] {cand.sub.name[:50]}: {str(e)[:90]}")
            continue
        db.execute("insert or replace into uploads values (?,?,?,?,?,?,?)",
                   (digest, str(cand.sub), cand.lang, status, sub_id, flags,
                    time.strftime("%Y-%m-%dT%H:%M:%S")))
        db.commit()
        sent_before.add(digest)
        if status == "duplicate":
            tally["duplicate"] += 1
            print(f"  dup  [{cand.lang}] {cand.sub.name[:50]} -> {sub_id}")
        else:
            tally["sent"] += 1
            print(f"  NEW  [{cand.lang}] {cand.sub.name[:50]} -> {sub_id} flags=[{flags}]")
        time.sleep(1.5)

    print(json.dumps(tally))
    return 1 if tally["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
