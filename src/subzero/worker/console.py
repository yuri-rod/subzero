"""Interactive console for a running Subzero worker.

Talks to the worker's HTTP API over loopback, so it works from the same host
without extra dependencies. Plain stdin prompts, matching the rest of the CLI.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KINDS = ("audit", "refetch", "resync", "embedded_translate", "rebuild",
         "recover_gaps", "repair")

STATE_COLOR = {
    "queued": "37", "running": "33", "done": "32", "failed": "31",
    "needs_review": "35", "paused": "36", "cancelled": "90",
}


class Console:
    def __init__(self, port: int, token: str):
        self.port = port
        self.token = token

    def _request(self, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, object]:
        url = f"http://127.0.0.1:{self.port}{path}"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, headers=headers, method=method, data=data)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
                try:
                    return resp.status, json.loads(raw)
                except ValueError:
                    return resp.status, raw
        except urllib.error.HTTPError as err:
            raw = err.read().decode("utf-8")
            try:
                return err.code, json.loads(raw)
            except ValueError:
                return err.code, raw
        except (urllib.error.URLError, OSError) as err:
            return 0, str(err)

    def _ok(self, code: int) -> bool:
        return 200 <= code < 300

    def _detail(self, payload: object) -> str:
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])
        return str(payload)

    def _print_state(self, state: str) -> str:
        color = STATE_COLOR.get(state, "0")
        if sys.stdout.isatty():
            return f"\x1b[{color}m{state:<12}\x1b[0m"
        return f"{state:<12}"

    # commands

    def cmd_status(self) -> None:
        code, body = self._request("/health")
        if not self._ok(code) or not isinstance(body, dict):
            print(f"  worker unreachable: {self._detail(body)}")
            return
        paused = body.get("paused", 0)
        runner = "running" if body.get("runner") else "idle"
        gpu = body.get("gpu")
        gpu_txt = f"{gpu} MB free" if gpu is not None else "n/a"
        print(f"  version   {body.get('version')}")
        print(f"  runner    {runner}   queue {body.get('queued', 0)}   paused {paused}")
        print(f"  model     {body.get('model')} ({body.get('whisperDevice')})")
        print(f"  translate {body.get('translation_provider')} / {body.get('translation_model')}")
        print(f"  auto      {body.get('auto')}   gpu {gpu_txt}")

    def cmd_jobs(self, args: list[str]) -> None:
        limit = 20
        if args and args[0].isdigit():
            limit = int(args[0])
        code, body = self._request(f"/jobs?limit={limit}")
        if not self._ok(code) or not isinstance(body, dict):
            print(f"  failed: {self._detail(body)}")
            return
        jobs = body.get("jobs", [])
        print(f"  {len(jobs)} jobs (downloads today {body.get('downloadsToday', 0)}/{body.get('budget', 0)})")
        for j in jobs:
            phase = f" {j.get('phase')}" if j.get("phase") else ""
            print(f"    {j['id'][:8]}  {self._print_state(j.get('state', ''))}"
                  f" {j.get('kind', ''):<16} {j.get('targetLang', ''):<6}"
                  f" {j.get('percent', 0):>3}%{phase}")
            msg = j.get("message")
            if msg and j.get("state") in ("failed", "needs_review"):
                print(f"        {msg[:120]}")

    def cmd_job(self, args: list[str]) -> None:
        if not args:
            print("  usage: job <id>")
            return
        code, body = self._request(f"/jobs/{args[0]}")
        if not self._ok(code) or not isinstance(body, dict):
            print(f"  failed: {self._detail(body)}")
            return
        for key in ("id", "itemId", "kind", "targetLang", "state", "outcome",
                    "phase", "percent", "attempts", "resultPath"):
            print(f"  {key:<10} {body.get(key)}")
        if body.get("message"):
            print(f"  message   {body['message']}")
        created = body.get("created")
        if created:
            print(f"  created   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created))}")

    def cmd_cancel(self, args: list[str]) -> None:
        if not args:
            print("  usage: cancel <id>")
            return
        code, body = self._request(f"/jobs/{args[0]}", method="DELETE")
        if self._ok(code) and isinstance(body, dict):
            print(f"  cancelled {args[0][:8]} -> {body.get('state')}")
        else:
            print(f"  failed: {self._detail(body)}")

    def cmd_resume(self, args: list[str]) -> None:
        if not args:
            print("  usage: resume <id>")
            return
        code, body = self._request(f"/jobs/{args[0]}/resume", method="POST")
        if self._ok(code) and isinstance(body, dict):
            print(f"  resumed {args[0][:8]} -> {body.get('state')}")
        else:
            print(f"  failed: {self._detail(body)}")

    def cmd_queue(self, args: list[str]) -> None:
        if len(args) < 3 or args[1] not in KINDS:
            print(f"  usage: queue <itemId> <kind> <targetLang>")
            print(f"  kinds: {', '.join(KINDS)}")
            return
        body = {"itemId": args[0], "kind": args[1], "targetLang": args[2]}
        code, resp = self._request("/jobs", method="POST", body=body)
        if self._ok(code) and isinstance(resp, dict):
            print(f"  enqueued {resp.get('id', '')[:8]} ({resp.get('kind')})")
        else:
            print(f"  failed: {self._detail(resp)}")

    def cmd_sweep(self) -> None:
        code, body = self._request("/sweep", method="POST")
        if self._ok(code) and isinstance(body, dict):
            print(f"  sweep enqueued {body.get('enqueued', 0)} jobs")
        else:
            print(f"  failed: {self._detail(body)}")

    def cmd_coverage(self, args: list[str]) -> None:
        lang = args[0] if args else "pt-BR"
        code, body = self._request(f"/coverage?lang={urllib.request.quote(lang)}")
        if not self._ok(code) or not isinstance(body, dict):
            print(f"  failed: {self._detail(body)}")
            return
        missing = body.get("missing", [])
        total = body.get("total", 0)
        print(f"  coverage {lang}: {total - len(missing)}/{total} ({len(missing)} missing)")
        for m in missing[:20]:
            print(f"    missing: {m.get('name') or m.get('itemId')}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    def cmd_audits(self) -> None:
        code, body = self._request("/sync/audits")
        if not self._ok(code) or not isinstance(body, dict):
            print(f"  failed: {self._detail(body)}")
            return
        audits = body.get("audits", [])
        print(f"  {len(audits)} audits (audit_only={body.get('auditOnly', False)})")
        for a in audits[:20]:
            print(f"    {a.get('status', ''):<8} {a.get('lang', ''):<6} {Path(a.get('video', '')).name}")

    def cmd_media(self, args: list[str]) -> None:
        if not args:
            print("  usage: media <itemId>")
            return
        code, body = self._request(f"/media/{args[0]}")
        if not self._ok(code) or not isinstance(body, dict):
            print(f"  failed: {self._detail(body)}")
            return
        print(f"  {body.get('name')}  [{body.get('container')}]  audio={body.get('audioLang')}")
        for s in body.get("embedded", []):
            where = "external" if s.get("external") else f"stream {s.get('index')}"
            print(f"    {s.get('lang', '?')} {s.get('codec', '')} {where}")
        for side in body.get("sidecars", []):
            print(f"    sidecar {Path(side).name}")

    def cmd_search(self, args: list[str]) -> None:
        if not args:
            print("  usage: search <itemId> [query]")
            return
        body = {"itemId": args[0], "query": args[1] if len(args) > 1 else None, "langs": []}
        code, resp = self._request("/search", method="POST", body=body)
        if not self._ok(code) or not isinstance(resp, dict):
            print(f"  failed: {self._detail(resp)}")
            return
        for c in resp.get("candidates", [])[:20]:
            print(f"    {c.get('fileId')}  {c.get('lang', ''):<6} {c.get('downloads', 0):>5}d  {c.get('release', '')[:48]}")

    def cmd_shutdown(self) -> None:
        code, body = self._request("/shutdown", method="POST")
        if self._ok(code):
            print("  shutdown triggered")
        else:
            print(f"  failed: {self._detail(body)}")

    def cmd_watch(self) -> None:
        print("  live monitor (Ctrl+C to stop)")
        try:
            while True:
                if sys.stdout.isatty():
                    sys.stdout.write("\x1b[2J\x1b[H")
                code, body = self._request("/health")
                if not self._ok(code) or not isinstance(body, dict):
                    print("  worker unreachable")
                else:
                    print(f"  queue {body.get('queued', 0)}  paused {body.get('paused', 0)}"
                          f"  runner {'on' if body.get('runner') else 'idle'}"
                          f"  {time.strftime('%H:%M:%S')}")
                    self.cmd_jobs(["8"])
                time.sleep(3)
        except KeyboardInterrupt:
            print("  stopped")

    def help(self) -> None:
        print("  status               worker health and queue summary")
        print("  jobs [N]             recent jobs (default 20)")
        print("  job <id>             one job in detail")
        print("  cancel <id>          cancel a job")
        print("  resume <id>          resume a paused job")
        print("  queue <itemId> <kind> <lang>   enqueue a job")
        print("  sweep                trigger a library sweep")
        print("  coverage [lang]      subtitle coverage report")
        print("  audits               recent sync audits")
        print("  media <itemId>       media streams and sidecars")
        print("  search <itemId> [q]  OpenSubtitles candidates")
        print("  watch                live refresh of status and queue")
        print("  shutdown             graceful worker shutdown")
        print("  help                 this list")
        print("  quit                 exit")

    def run(self) -> int:
        print(f"subzero worker console (http://127.0.0.1:{self.port})")
        print("type 'help' for commands, 'quit' to leave")
        while True:
            try:
                raw = input("worker> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not raw:
                continue
            parts = raw.split()
            cmd, args = parts[0].lower(), parts[1:]
            if cmd in ("q", "quit", "exit"):
                return 0
            if cmd in ("help", "?", "h"):
                self.help()
            elif cmd in ("status", "health"):
                self.cmd_status()
            elif cmd == "jobs":
                self.cmd_jobs(args)
            elif cmd == "job":
                self.cmd_job(args)
            elif cmd == "cancel":
                self.cmd_cancel(args)
            elif cmd == "resume":
                self.cmd_resume(args)
            elif cmd == "queue":
                self.cmd_queue(args)
            elif cmd == "sweep":
                self.cmd_sweep()
            elif cmd == "coverage":
                self.cmd_coverage(args)
            elif cmd == "audits":
                self.cmd_audits()
            elif cmd == "media":
                self.cmd_media(args)
            elif cmd == "search":
                self.cmd_search(args)
            elif cmd in ("watch", "monitor"):
                self.cmd_watch()
            elif cmd in ("shutdown", "stop"):
                self.cmd_shutdown()
            else:
                print(f"  unknown command '{cmd}' (try 'help')")


def run_console(port: int, token: str) -> int:
    return Console(port, token).run()
