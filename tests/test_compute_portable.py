import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from subzero import compute
from subzero.compute import _load_policy


@pytest.fixture(autouse=True)
def local_policy(tmp_path, monkeypatch, isolate_compute_policy):
    monkeypatch.delenv('SUBZERO_COMPUTE_CONFIG', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(compute, '_load_policy', _load_policy)


@pytest.fixture
def windows_os(monkeypatch):
    monkeypatch.setattr(compute, 'os', SimpleNamespace(
        name='nt', environ=os.environ, open=os.open, O_RDONLY=os.O_RDONLY))


@pytest.mark.parametrize('kind', ['vision', 'whisper', 'ollama'])
@pytest.mark.parametrize('windows', [False, True])
def test_missing_default_policy_preserves_portable_behavior(kind, windows, request):
    if windows:
        request.getfixturevalue('windows_os')
    assert compute.compute_lease_fd() is None
    with compute.compute_phase(kind) as strict:
        assert strict is False
        assert compute.compute_lease_fd() is None


def test_explicit_missing_policy_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('SUBZERO_COMPUTE_CONFIG', str(tmp_path / 'missing.json'))
    with pytest.raises(RuntimeError, match='policy'):
        with compute.compute_phase('vision'):
            pytest.fail('compute must not start')


@pytest.mark.parametrize('selection', ['explicit_missing', 'explicit_existing', 'default'])
@pytest.mark.parametrize('operation', ['phase', 'lease'])
def test_windows_policy_fails_before_posix_file_or_lock_operations(
        tmp_path, monkeypatch, windows_os, selection, operation):
    path = tmp_path / '.config/subzero/compute.json'
    if selection != 'explicit_missing':
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    if selection != 'default':
        monkeypatch.setenv('SUBZERO_COMPUTE_CONFIG', str(path))
    with pytest.raises(RuntimeError, match='unsupported'):
        if operation == 'lease':
            compute.compute_lease_fd()
        else:
            with compute.compute_phase('vision'):
                pytest.fail('unsupported policy must not enter compute')
