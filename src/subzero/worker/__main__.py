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
    if custom:
        p = Path(custom).expanduser().resolve()
        if p.exists():
            return p
    for candidate in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
        Path.home() / ".config/subzero/.env",
        Path.home() / ".config/srtworker/.env",
    ]:
        if candidate.exists():
            return candidate
    return None


def run_worker_cmd(action: str = "serve", env_file: str | Path | None = None, port: int | None = None) -> int:
    env_path = find_env_file(env_file)
    env = read_env_file(env_path) if env_path else {}
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
        if arg in ("--env", "-e") and idx + 1 < len(args):
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
