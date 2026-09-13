import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from .api import create_app
from .config import Config, read_env_file


def _request(path: str, method: str = "GET", token: str = "", port: int = 8787) -> tuple[int, str]:
    url = f"http://127.0.0.1:{port}{path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")
    except Exception as err:
        return 0, str(err)


def find_env_file(custom: str | Path | None = None) -> Path | None:
    if custom is not None:
        p = Path(custom).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"Environment file does not exist: {p}")
        if not p.is_file():
            raise ValueError(f"Environment path is not a file: {p}")
        return p
    for candidate in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
        Path.home() / ".config/subzero/.env",
    ]:
        if candidate.exists():
            return candidate
    return None


def run_worker_cmd(action: str = "serve", env_file: str | Path | None = None, port: int | None = None) -> int:
    try:
        env_path = find_env_file(env_file)
        env = read_env_file(env_path, required=env_file is not None) if env_path else {}
    except (OSError, ValueError) as err:
        print(f"subzero worker: {err}", file=sys.stderr)
        return 1
    env = {**env, **os.environ}
    effective_port = port or int(env.get("PORT", "8787"))
    token = env.get("BEARER_TOKEN", "")

    cmd = (action or "serve").lower()
    if cmd in ("start", "--start"):
        if sys.platform == "darwin":
            import subprocess
            res = subprocess.run(["launchctl", "start", "com.yuri.srt-worker"], capture_output=True)
            if res.returncode == 0:
                print("subzero worker: started via launchd")
                return 0
        cmd = "serve"
    if cmd in ("stop", "shutdown", "--stop"):
        code, body = _request("/shutdown", method="POST", token=token, port=effective_port)
        if code == 200:
            print("subzero worker: shutdown triggered")
            return 0
        print(f"subzero worker: failed to shutdown (code {code}): {body}", file=sys.stderr)
        return 1
    if cmd in ("status", "--status"):
        code, body = _request("/health", method="GET", token=token, port=effective_port)
        if code == 200:
            print(f"subzero worker running: {body}")
            return 0
        print(f"subzero worker not running or unreachable ({body})", file=sys.stderr)
        return 1
    if cmd in ("contribute", "--contribute"):
        from .contribute import main as contribute_main
        contribute_argv = []
        if env_path:
            contribute_argv.extend(["--env", str(env_path)])
        return contribute_main(contribute_argv)
    if cmd in ("jobs", "--jobs"):
        code, body = _request("/jobs", method="GET", token=token, port=effective_port)
        if code == 200:
            try:
                data = json.loads(body)
                jobs = data.get("jobs", [])
                dl = data.get("downloadsToday", 0)
                budget = data.get("budget", 0)
                print(f"subzero worker: {len(jobs)} jobs (downloads today: {dl}/{budget})")
                for j in jobs:
                    phase = f" [{j['phase']}]" if j.get("phase") else ""
                    print(f"  {j['id'][:8]} {j['kind']:<10} {j['targetLang']:<6} {j['state']:<8} {j['percent']:>3}%{phase}")
            except Exception:
                print(body)
            return 0
        print(f"subzero worker: failed to fetch jobs (code {code}): {body}", file=sys.stderr)
        return 1
    if cmd in ("sweep", "--sweep"):
        code, body = _request("/sweep", method="POST", token=token, port=effective_port)
        if code == 200:
            try:
                data = json.loads(body)
                print(f"subzero worker: sweep enqueued {data.get('enqueued', 0)} jobs")
            except Exception:
                print(body)
            return 0
        print(f"subzero worker: failed to trigger sweep (code {code}): {body}", file=sys.stderr)
        return 1
    if cmd in ("coverage", "--coverage"):
        code, body = _request("/coverage", method="GET", token=token, port=effective_port)
        if code == 200:
            try:
                data = json.loads(body)
                missing = data.get("missing", [])
                total = data.get("total", 0)
                print(f"subzero worker: coverage for {data.get('lang', 'pt-BR')}: {total - len(missing)}/{total} ({len(missing)} missing)")
                for m in missing[:20]:
                    print(f"  missing: {m.get('name') or m.get('itemId')}")
                if len(missing) > 20:
                    print(f"  ... and {len(missing) - 20} more")
            except Exception:
                print(body)
            return 0
        print(f"subzero worker: failed to fetch coverage (code {code}): {body}", file=sys.stderr)
        return 1
    if cmd in ("audits", "--audits"):
        code, body = _request("/sync/audits", method="GET", token=token, port=effective_port)
        if code == 200:
            try:
                data = json.loads(body)
                audits = data.get("audits", [])
                print(f"subzero worker: {len(audits)} recent audits (audit_only={data.get('auditOnly', False)})")
                for a in audits[:20]:
                    print(f"  {a.get('status', ''):<8} {a.get('lang', ''):<6} {Path(a.get('video', '')).name}")
            except Exception:
                print(body)
            return 0
        print(f"subzero worker: failed to fetch audits (code {code}): {body}", file=sys.stderr)
        return 1
    if cmd in ("triage", "--triage"):
        from .applefm import AppleFM
        from .notify import Notifier
        from .triage import cluster, summarize
        code, body = _request("/jobs?limit=200", method="GET", token=token, port=effective_port)
        if code != 200:
            print(f"subzero worker: failed to fetch jobs (code {code}): {body}", file=sys.stderr)
            return 1
        try:
            jobs = json.loads(body).get("jobs", [])
        except Exception:
            print(body)
            return 1
        groups = cluster(jobs if isinstance(jobs, list) else [])
        fm_url = env.get("APPLEFM_URL", "http://127.0.0.1:1976").rstrip("/")
        try:
            digest = summarize(groups, AppleFM(fm_url, timeout=60))
        except Exception:
            from .triage import template_digest
            digest = template_digest(groups)
        print(digest)
        topic = env.get("NTFY_TOPIC", "")
        if topic:
            Notifier(env.get("NTFY_URL", "https://ntfy.sh"), topic).digest(digest)
            print(f"subzero worker: triage sent to ntfy topic {topic}")
        return 0

    try:
        cfg = Config.load(env)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=effective_port, log_level="info"))
    app.state.server = server
    server.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    env_file = None
    port = None
    action = "serve"
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg in ("--env", "-e"):
            if idx + 1 == len(args):
                print(f"subzero worker: {arg} requires a file path", file=sys.stderr)
                return 1
            env_file = args[idx + 1]
            idx += 2
        elif arg in ("--port", "-p") and idx + 1 < len(args):
            port = int(args[idx + 1])
            idx += 2
        elif not arg.startswith("-"):
            action = arg
            idx += 1
        else:
            idx += 1
    return run_worker_cmd(action=action, env_file=env_file, port=port)


if __name__ == "__main__":
    raise SystemExit(main())
