import json

from subzero.worker.console import Console, KINDS, run_console


def make_console(monkeypatch, responses):
    console = Console(8787, "tok")

    def fake_request(path, method="GET", body=None):
        return responses.get((path, method), (404, {"detail": "not found"}))

    monkeypatch.setattr(console, "_request", fake_request)
    return console


def test_console_status(monkeypatch, capsys):
    console = make_console(monkeypatch, {
        ("/health", "GET"): (200, {"version": "1.15.0", "gpu": None, "model": "m", "whisperDevice": "mlx",
                                   "translation_provider": "ollama", "translation_model": "x", "paused": 1,
                                   "auto": False, "queued": 3, "runner": True}),
    })
    console.cmd_status()
    out = capsys.readouterr().out
    assert "1.15.0" in out
    assert "queue 3" in out
    assert "paused 1" in out


def test_console_jobs(monkeypatch, capsys):
    data = {"jobs": [{"id": "abc12345def", "kind": "repair", "targetLang": "pt-BR", "state": "running",
                      "percent": 50, "phase": "scanning"}], "downloadsToday": 3, "budget": 15}
    console = make_console(monkeypatch, {("/jobs?limit=5", "GET"): (200, data)})
    console.cmd_jobs(["5"])
    out = capsys.readouterr().out
    assert "repair" in out
    assert "50%" in out


def test_console_queue_requires_valid_kind(monkeypatch, capsys):
    console = make_console(monkeypatch, {})
    console.cmd_queue(["item", "bogus", "pt-BR"])
    assert "usage: queue" in capsys.readouterr().out


def test_console_queue_enqueues(monkeypatch, capsys):
    console = make_console(monkeypatch, {
        ("/jobs", "POST"): (200, {"id": "deadbeef0001", "kind": "repair", "targetLang": "pt-BR"}),
    })
    console.cmd_queue(["item", "repair", "pt-BR"])
    out = capsys.readouterr().out
    assert "enqueued deadbeef" in out


def test_console_cancel_and_resume(monkeypatch, capsys):
    console = make_console(monkeypatch, {
        ("/jobs/j1", "DELETE"): (200, {"id": "j1", "state": "cancelled"}),
        ("/jobs/j2/resume", "POST"): (200, {"id": "j2", "state": "queued"}),
    })
    console.cmd_cancel(["j1"])
    assert "cancelled" in capsys.readouterr().out
    console.cmd_resume(["j2"])
    assert "resumed" in capsys.readouterr().out


def test_console_coverage(monkeypatch, capsys):
    data = {"lang": "pt-BR", "total": 10, "missing": [{"itemId": "i1", "name": "Ep"}]}
    console = make_console(monkeypatch, {("/coverage?lang=pt-BR", "GET"): (200, data)})
    console.cmd_coverage([])
    assert "9/10" in capsys.readouterr().out


def test_console_run_quits(monkeypatch, capsys):
    console = make_console(monkeypatch, {})
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
    assert console.run() == 0


def test_console_run_dispatches(monkeypatch, capsys):
    console = make_console(monkeypatch, {
        ("/health", "GET"): (200, {"version": "v", "gpu": None, "model": "m", "whisperDevice": "mlx",
                                   "translation_provider": "ollama", "translation_model": "x", "paused": 0,
                                   "auto": False, "queued": 0, "runner": True}),
    })
    inputs = iter(["status", "nonsense", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert console.run() == 0
    out = capsys.readouterr().out
    assert "unknown command 'nonsense'" in out


def test_run_console_entry(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
    assert run_console(8787, "") == 0


def test_kinds_nonempty():
    assert "repair" in KINDS
    assert "rebuild" in KINDS
