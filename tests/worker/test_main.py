from pathlib import Path

import pytest

from subzero.worker import __main__ as worker_main


@pytest.fixture
def config_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    install = tmp_path / "install"
    for folder in (home, cwd, install):
        folder.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(worker_main, "__file__", str(install / "src/subzero/worker/__main__.py"))
    return home, cwd, install


def test_obsolete_worker_config_requires_explicit_path(config_paths):
    home, _, _ = config_paths
    legacy = home / ".config/srtworker/.env"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("PORT=8788\n")

    assert worker_main.find_env_file() is None
    assert worker_main.find_env_file(legacy) == legacy


def test_worker_config_discovery_uses_current_locations(config_paths):
    home, cwd, install = config_paths
    current = home / ".config/subzero/.env"
    current.parent.mkdir(parents=True)
    current.write_text("PORT=8787\n")
    assert worker_main.find_env_file() == current

    installed = install / ".env"
    installed.write_text("PORT=8788\n")
    assert worker_main.find_env_file() == installed

    local = cwd / ".env"
    local.write_text("PORT=8789\n")
    assert worker_main.find_env_file() == local
    assert worker_main.find_env_file(current) == current


def test_explicit_missing_config_does_not_fall_back_to_discovered_file(config_paths):
    _, cwd, _ = config_paths
    (cwd / ".env").write_text("OLLAMA_MODEL=another-model\n")
    missing = cwd / "missing.env"

    with pytest.raises(ValueError, match="Environment file does not exist"):
        worker_main.find_env_file(missing)


@pytest.mark.parametrize("explicit", [".", ""])
def test_explicit_config_must_name_a_file(config_paths, explicit):
    with pytest.raises(ValueError, match="Environment path is not a file"):
        worker_main.find_env_file(explicit)


@pytest.mark.parametrize("action", ["serve", "start", "status", "stop", "contribute"])
def test_invalid_explicit_config_stops_before_reading_another_file(config_paths, monkeypatch, capsys, action):
    _, cwd, _ = config_paths
    (cwd / ".env").write_text("OLLAMA_MODEL=another-model\n")
    missing = cwd / "missing.env"

    def unexpected_read(path):
        raise AssertionError("An invalid explicit config must stop discovery")

    monkeypatch.setattr(worker_main, "read_env_file", unexpected_read)
    assert worker_main.run_worker_cmd(action, env_file=missing) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Environment file does not exist" in captured.err
    assert str(missing) in captured.err


@pytest.mark.parametrize("option", ["--env", "-e"])
def test_explicit_config_option_requires_a_path(monkeypatch, capsys, option):
    def unexpected_run(**kwargs):
        raise AssertionError("A missing config argument must stop before worker actions")

    monkeypatch.setattr(worker_main, "run_worker_cmd", unexpected_run)
    assert worker_main.main(["status", option]) == 1
    assert "requires a file path" in capsys.readouterr().err


def test_explicit_config_disappearing_before_read_stops_worker_action(config_paths, monkeypatch, capsys):
    _, cwd, _ = config_paths
    explicit = cwd / "selected.env"
    explicit.write_text("PORT=43210\n")
    read = worker_main.read_env_file

    def remove_before_read(path, **kwargs):
        Path(path).unlink()
        return read(path, **kwargs)

    def unexpected_request(*args, **kwargs):
        raise AssertionError("A missing explicit config must stop before an HTTP request")

    monkeypatch.setattr(worker_main, "read_env_file", remove_before_read)
    monkeypatch.setattr(worker_main, "_request", unexpected_request)
    assert worker_main.run_worker_cmd("status", env_file=explicit) == 1
    assert repr(str(explicit)) in capsys.readouterr().err


def test_run_worker_cmd_jobs(monkeypatch, capsys):
    import json
    calls = []
    fake_jobs_data = {
        "jobs": [
            {"id": "abc12345def", "kind": "repair", "targetLang": "pt-BR", "state": "running", "percent": 50, "phase": "scanning"}
        ],
    }

    def fake_request(path, method="GET", token="", port=8787):
        calls.append((path, method, token, port))
        return 200, json.dumps(fake_jobs_data)

    monkeypatch.setattr(worker_main, "_request", fake_request)
    assert worker_main.run_worker_cmd("jobs", port=8787) == 0
    assert calls[0][0] == "/jobs"
    assert calls[0][1] == "GET"
    assert calls[0][3] == 8787
    out = capsys.readouterr().out
    assert "1 jobs" in out
    assert "abc12345 repair     pt-BR  running   50% [scanning]" in out

    monkeypatch.setattr(worker_main, "_request", lambda *a, **kw: (500, "server error"))
    assert worker_main.run_worker_cmd("jobs") == 1
    assert "failed to fetch jobs (code 500)" in capsys.readouterr().err


def test_run_worker_cmd_sweep(monkeypatch, capsys):
    import json
    monkeypatch.setattr(worker_main, "_request", lambda path, method="GET", token="", port=8787: (
        200, json.dumps({"enqueued": 4})
    ))
    assert worker_main.run_worker_cmd("sweep") == 0
    assert "sweep enqueued 4 jobs" in capsys.readouterr().out

    monkeypatch.setattr(worker_main, "_request", lambda *a, **kw: (401, "unauthorized"))
    assert worker_main.run_worker_cmd("sweep") == 1
    assert "failed to trigger sweep (code 401)" in capsys.readouterr().err


def test_run_worker_cmd_coverage(monkeypatch, capsys):
    import json
    data = {
        "lang": "pt-BR",
        "total": 10,
        "missing": [{"itemId": "item1", "name": "Survivor S47E01"}]
    }
    monkeypatch.setattr(worker_main, "_request", lambda path, method="GET", token="", port=8787: (
        200, json.dumps(data)
    ))
    assert worker_main.run_worker_cmd("coverage") == 0
    out = capsys.readouterr().out
    assert "coverage for pt-BR: 9/10 (1 missing)" in out
    assert "missing: Survivor S47E01" in out

    monkeypatch.setattr(worker_main, "_request", lambda *a, **kw: (502, "bad gateway"))
    assert worker_main.run_worker_cmd("coverage") == 1
    assert "failed to fetch coverage (code 502)" in capsys.readouterr().err


def test_run_worker_cmd_audits(monkeypatch, capsys):
    import json
    data = {
        "auditOnly": False,
        "audits": [{"status": "pass", "lang": "pt-BR", "video": "/media/episode.mkv"}]
    }
    monkeypatch.setattr(worker_main, "_request", lambda path, method="GET", token="", port=8787: (
        200, json.dumps(data)
    ))
    assert worker_main.run_worker_cmd("audits") == 0
    out = capsys.readouterr().out
    assert "1 recent audits (audit_only=False)" in out
    assert "pass     pt-BR  episode.mkv" in out

    monkeypatch.setattr(worker_main, "_request", lambda *a, **kw: (404, "not found"))
    assert worker_main.run_worker_cmd("audits") == 1
    assert "failed to fetch audits (code 404)" in capsys.readouterr().err

