"""Fill missing subtitles from OpenSubtitles and verify their sync afterwards.

Runs from a LaunchAgent after the OpenSubtitles daily quota resets. Queues
opensubtitles jobs straight into the worker database, waits for the runner to
drain them, then checks each new sidecar against the spoken dialogue and
resyncs the ones that came from a different release. With --retire the agent
boots itself out and deletes its plist once nothing is missing anymore.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time

from subzero.worker.config import Config, read_env_file
from subzero.worker.jellyfin import JellyfinClient
from subzero.worker.jobs import JobStore
from subzero.worker.opensubs import OpenSubtitles
from subzero.worker.watch import Watcher, has_language

REPO = Path(__file__).resolve().parent.parent
REFERENCE_CACHE = Path.home() / '.cache/subzero/references'
BACKUP_DIR = Path.home() / '.cache/subzero/backups/refill'
LAUNCH_LABEL = 'com.yuri.subzero-refill'
LAUNCH_PLIST = Path.home() / 'Library/LaunchAgents' / f'{LAUNCH_LABEL}.plist'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', default='', help='Jellyfin series name; empty means the whole library')
    parser.add_argument('--lang', default='pt-BR')
    parser.add_argument('--limit', type=int, default=20, help='downloads per run')
    parser.add_argument('--wait', type=int, default=3600, help='seconds to wait for the runner')
    parser.add_argument('--retire', action='store_true',
                        help='boot the LaunchAgent out and delete it when nothing is missing')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def retire_agent():
    """Remove the agent from a detached shell so this run can exit first."""
    print(f'nothing left to refill; retiring {LAUNCH_LABEL}')
    command = f'sleep 1; /bin/launchctl bootout gui/{os.getuid()}/{LAUNCH_LABEL} 2>/dev/null;'
    command += f' rm -f {shlex.quote(str(LAUNCH_PLIST))}'
    subprocess.Popen(['/bin/sh', '-c', command], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def missing_media(jelly, series, lang):
    found = []
    for item in jelly.all_items():
        if series and item.get('SeriesName') != series:
            continue
        media = jelly.media(item['Id'])
        if has_language(media, lang):
            continue
        found.append(media)
    return found


def wait_for_jobs(store, ids, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(job.id in ids for job in store.active()):
            return True
        time.sleep(15)
    return False


def check_sync(results):
    from subzero.reference import build_reference, verify_text
    from subzero.sync import auto_sync_file

    for media, sub in results:
        report = verify_text(sub.read_text(encoding='utf-8-sig'),
                             build_reference(media.path, REFERENCE_CACHE))
        if report.status == 'pass':
            print(f'{sub.name}: in sync')
            continue
        print(f'{sub.name}: {report.status} ({report.reason}), resyncing')
        try:
            result = auto_sync_file(media.path, sub, backup_dir=BACKUP_DIR)
        except Exception as err:
            print(f'{sub.name}: sync failed: {err}')
            continue
        again = verify_text(sub.read_text(encoding='utf-8-sig'),
                            build_reference(media.path, REFERENCE_CACHE))
        print(f'{sub.name}: {result.method} -> {again.status}')


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.load(read_env_file(REPO / '.env', required=True))
    jelly = JellyfinClient(cfg.jellyfin_url, cfg.jellyfin_key, bare_lang=cfg.bare_lang)
    opensubs = OpenSubtitles(cfg.opensubtitles_key, username=cfg.opensubtitles_user,
                             password=cfg.opensubtitles_password)
    opensubs.login()
    account = opensubs.account()
    print(f'OpenSubtitles quota: {account.remaining}/{account.allowed} ({account.level})')

    missing = missing_media(jelly, args.series, args.lang)
    if not missing:
        print('nothing missing')
        if args.retire:
            retire_agent()
        return 0
    if account.remaining <= 0 and not args.dry_run:
        print(f'{len(missing)} items missing but the quota is exhausted; the next run picks them up')
        return 0

    store = JobStore(cfg.db_path)
    watcher = Watcher(jelly, store, opensubs, cfg.state_path, [args.lang],
                      fallback_langs=cfg.fallback_langs, translate_from=cfg.translate_from,
                      excluded_paths=cfg.excluded_paths)
    pending = {(job.item_id, job.kind) for job in store.active()}
    limit = min(args.limit, account.remaining) if account.remaining > 0 else args.limit
    queued = []
    selected = 0
    for media in missing:
        if selected >= limit:
            break
        if (media.item_id, 'opensubtitles') in pending:
            continue
        file_id = watcher.pick_candidate(media, args.lang)
        if not file_id:
            print(f'{media.name}: no candidate')
            continue
        sub = Path(media.path).with_suffix(f'.{args.lang}.srt')
        selected += 1
        if args.dry_run:
            print(f'{media.name}: would download {file_id} -> {sub.name}')
            continue
        job = store.enqueue(media.item_id, 'opensubtitles', args.lang, file_id)
        queued.append((media, job.id, sub))
        print(f'{media.name}: queued {job.id[:8]} file {file_id}')

    if args.dry_run or not queued:
        return 0
    if not wait_for_jobs(store, {job_id for _, job_id, _ in queued}, args.wait):
        print('timed out waiting for the runner')
        return 1

    done = []
    for media, job_id, sub in queued:
        job = store.get(job_id)
        if job and job.state == 'done' and sub.exists():
            done.append((media, sub))
        else:
            state = job.state if job else 'missing'
            print(f'{media.name}: {state} {job.message if job else ""}'.strip())
    if done:
        print(f'\nchecking sync of {len(done)} new sidecars')
        check_sync(done)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
