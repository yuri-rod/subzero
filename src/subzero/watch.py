"""Keep a directory clean as subtitles arrive."""

from __future__ import annotations

import concurrent.futures as futures
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .core import Options, fix_file

SETTLE = 120        # leave a file alone until it has stopped being written
INTERVAL = 300
WORKERS = 6


@dataclass
class Watcher:
    roots: list[Path]
    opts: Options = field(default_factory=Options)
    state_path: Path | None = None
    backup_dir: str | None = None
    pattern: str = "*.srt"
    skip: str = ""
    settle: int = SETTLE
    workers: int = WORKERS
    on_event: Callable[[str], None] | None = None

    def __post_init__(self):
        self.state = self._load()

    def _load(self) -> dict:
        if not self.state_path:
            return {}
        try:
            return json.loads(Path(self.state_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        if not self.state_path:
            return
        tmp = f"{self.state_path}.part"
        Path(tmp).write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _say(self, msg: str) -> None:
        if self.on_event:
            self.on_event(msg)

    def candidates(self, now: float):
        import fnmatch

        for root in self.roots:
            root = Path(root)
            if not root.exists():
                continue
            for p in root.rglob(self.pattern):
                if self.skip and fnmatch.fnmatch(p.name, self.skip):
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                # settle=0 means no quiet period at all. Without the guard a
                # file whose timestamp sits a hair ahead of our clock, which
                # happens on Windows, goes negative here and is skipped forever.
                if self.settle and now - st.st_mtime < self.settle:
                    continue
                key = str(p)
                # size as well as mtime: on mtime alone the tool's own rewrite
                # looks like a fresh change and every sweep redoes the library.
                # mtime stays a float: truncating to whole seconds hid any edit
                # that landed in the same second as the previous one, and every
                # filesystem we run on keeps sub-second resolution.
                stamp = [st.st_mtime, st.st_size]
                if self.state.get(key) == stamp:
                    continue
                yield p, key

    def sweep(self) -> tuple[int, int]:
        todo = list(self.candidates(time.time()))
        if not todo:
            return 0, 0
        fixed = failed = 0
        with futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            jobs = {
                pool.submit(fix_file, p, self.opts, self.backup_dir): (p, key)
                for p, key in todo
            }
            for fut in futures.as_completed(jobs):
                p, key = jobs[fut]
                try:
                    res = fut.result()
                except Exception as e:                      # noqa: BLE001
                    failed += 1
                    self._say(f"failed {p.name}: {str(e)[:120]}")
                    continue
                if res is None:
                    failed += 1
                    self._say(f"failed {p.name}: no cues parsed")
                    continue
                if res.changed:
                    fixed += 1
                    self._say(
                        f"fixed {p.name} cues={res.cues} "
                        f"sdh-{res.dropped} rewrap={res.rewrapped}"
                    )
                try:
                    st = p.stat()
                    self.state[key] = [st.st_mtime, st.st_size]
                except OSError:
                    self.state.pop(key, None)
        self._save()
        return fixed, failed

    def run(self, interval: int = INTERVAL, once: bool = False) -> None:
        while True:
            fixed, failed = self.sweep()
            if fixed or failed:
                self._say(f"sweep: {fixed} fixed, {failed} failed")
            if once:
                return
            time.sleep(interval)
