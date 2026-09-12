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
