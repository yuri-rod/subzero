import json
import multiprocessing
import os
import plistlib
import threading
import time
from pathlib import Path

import pytest

from subzero import compute


@pytest.fixture
def policy(tmp_path, monkeypatch):
    executable = tmp_path / 'ollama'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    agent = tmp_path / 'com.example.ollama.plist'
    agent.write_bytes(plistlib.dumps({'Label': 'com.example.ollama',
                                    'ProgramArguments': [str(executable), 'serve'],
                                    'KeepAlive': True,
                                    'EnvironmentVariables': {'OLLAMA_HOST': '127.0.0.1:11434'}}))
    config = tmp_path / 'compute.json'
    config.write_text(json.dumps({'ollama_url': 'http://127.0.0.1:11434',
                                 'ollama_launch_agent': str(agent),
                                 'lock_path': str(tmp_path / 'compute.lock'),
                                 'startup_timeout': 2, 'shutdown_timeout': 2}))
    monkeypatch.setenv('SUBZERO_COMPUTE_CONFIG', str(config))
    return config


@pytest.fixture
def lifecycle(monkeypatch):
    events = []

    class Lifecycle:
        def __init__(self, policy, lock):
            pass

        def stop(self):
            events.append('stopped')

        def start(self):
            events.append('ready')

    monkeypatch.setattr(compute, '_Lifecycle', Lifecycle)
    return events


def test_missing_default_policy_preserves_portable_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv('SUBZERO_COMPUTE_CONFIG', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with compute.compute_phase('vision') as strict:
        assert strict is False


def test_explicit_missing_policy_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('SUBZERO_COMPUTE_CONFIG', str(tmp_path / 'missing.json'))
    with pytest.raises(RuntimeError, match='policy'):
        with compute.compute_phase('vision'):
            pytest.fail('compute must not start')


def test_phase_order_and_same_thread_nesting(policy, lifecycle):
    with compute.compute_phase('ollama', ollama_url='http://127.0.0.1:11434/') as strict:
        assert strict is True
        assert lifecycle == ['stopped', 'ready']
        with compute.compute_phase('ollama') as nested:
            assert nested is True
            lifecycle.append('nested inference')
    assert lifecycle == ['stopped', 'ready', 'nested inference', 'stopped']


def test_non_ollama_phase_stops_service_without_starting_it(policy, lifecycle):
    with compute.compute_phase('vision'):
        lifecycle.append('vision exited')
    with compute.compute_phase('whisper'):
        lifecycle.append('whisper exited')
    assert lifecycle == ['stopped', 'vision exited', 'stopped', 'whisper exited']


def test_nested_handoff_and_url_mismatch_are_rejected(policy, lifecycle):
    with compute.compute_phase('vision'):
        with pytest.raises(RuntimeError, match='nested'):
            with compute.compute_phase('ollama'):
                pytest.fail('different compute cannot start')
    with pytest.raises(RuntimeError, match='URL'):
        with compute.compute_phase('ollama', ollama_url='http://127.0.0.1:11435'):
            pytest.fail('unmanaged model cannot start')


def test_exception_still_stops_service(policy, lifecycle):
    with pytest.raises(KeyboardInterrupt):
        with compute.compute_phase('ollama'):
            raise KeyboardInterrupt()
    assert lifecycle == ['stopped', 'ready', 'stopped']


def test_shutdown_failure_is_not_silenced(policy, monkeypatch):
    class Lifecycle:
        def __init__(self, policy, lock):
            self.calls = 0

        def start(self):
            pass

        def stop(self):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('runner still alive')

    monkeypatch.setattr(compute, '_Lifecycle', Lifecycle)
    with pytest.raises(RuntimeError, match='runner still alive'):
        with compute.compute_phase('ollama'):
            pass


def test_failed_start_is_followed_by_shutdown(policy, monkeypatch):
    events = []

    class Lifecycle:
        def __init__(self, policy, lock):
            pass

        def start(self):
            events.append('start failed')
            raise RuntimeError('readiness failed')

        def stop(self):
            events.append('stopped')

    monkeypatch.setattr(compute, '_Lifecycle', Lifecycle)
    with pytest.raises(RuntimeError, match='readiness failed'):
        with compute.compute_phase('ollama'):
            pytest.fail('compute must not start')
    assert events == ['stopped', 'start failed', 'stopped']


def test_competing_threads_wait_until_shutdown_completes(policy, lifecycle):
    entered = threading.Event()
    release = threading.Event()
    other = threading.Event()
    failures = []

    def first():
        try:
            with compute.compute_phase('ollama'):
                entered.set()
                assert release.wait(2)
        except BaseException as err:
            failures.append(err)

    def second():
        try:
            with compute.compute_phase('vision'):
                other.set()
        except BaseException as err:
            failures.append(err)

    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    assert entered.wait(2)
    b.start()
    assert not other.wait(.1)
    release.set()
    a.join(3)
    b.join(3)
    assert not failures
    assert other.is_set()
    assert lifecycle == ['stopped', 'ready', 'stopped', 'stopped']


def _process_phase(config, log, ready, release, name):
    os.environ['SUBZERO_COMPUTE_CONFIG'] = config

    class Lifecycle:
        def __init__(self, policy, lock):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    compute._Lifecycle = Lifecycle
    with compute.compute_phase('vision'):
        with open(log, 'a') as handle:
            handle.write(name + ' enter\n')
        ready.set()
        if release is not None:
            assert release.wait(3)
        with open(log, 'a') as handle:
            handle.write(name + ' exit\n')


def test_competing_processes_share_one_phase_lock(policy, tmp_path):
    ctx = multiprocessing.get_context('spawn')
    a_ready, b_ready, release = ctx.Event(), ctx.Event(), ctx.Event()
    log = tmp_path / 'phases.log'
    a = ctx.Process(target=_process_phase, args=(str(policy), str(log), a_ready, release, 'a'))
    b = ctx.Process(target=_process_phase, args=(str(policy), str(log), b_ready, None, 'b'))
    try:
        a.start()
        assert a_ready.wait(3)
        b.start()
        assert not b_ready.wait(.2)
        release.set()
        a.join(4)
        b.join(4)
        assert (a.exitcode, b.exitcode) == (0, 0)
        assert log.read_text().splitlines() == ['a enter', 'a exit', 'b enter', 'b exit']
    finally:
        for process in (a, b):
            if process.is_alive():
                process.terminate()
                process.join(2)


@pytest.mark.parametrize('field,value', [
    ('ollama_url', 'http://example.com:11434'),
    ('ollama_url', 'http://127.0.0.1:11434/path'),
    ('ollama_url', 'http://user:secret@127.0.0.1:11434'),
    ('lock_path', 'relative.lock'),
    ('shutdown_timeout', 0),
    ('startup_timeout', float('nan')),
])
def test_unsafe_policy_never_enters_compute(policy, lifecycle, field, value):
    config = json.loads(policy.read_text())
    config[field] = value
    policy.write_text(json.dumps(config))
    with pytest.raises(RuntimeError):
        with compute.compute_phase('vision'):
            pytest.fail('invalid policy cannot start compute')
    assert not lifecycle


def test_agent_must_be_an_owned_ollama_serve_job(policy, lifecycle):
    config = json.loads(policy.read_text())
    agent = Path(config['ollama_launch_agent'])
    job = plistlib.loads(agent.read_bytes())
    job['ProgramArguments'] = ['/bin/sh', '-c', 'ollama serve']
    agent.write_bytes(plistlib.dumps(job))
    with pytest.raises(RuntimeError, match='Ollama serve'):
        with compute.compute_phase('vision'):
            pytest.fail('arbitrary service must not be managed')
    assert not lifecycle


def test_symlink_lock_cannot_be_used(policy, lifecycle, tmp_path):
    config = json.loads(policy.read_text())
    victim = tmp_path / 'untouched'
    victim.write_text('keep')
    Path(config['lock_path']).symlink_to(victim)
    with pytest.raises(RuntimeError, match='lock'):
        with compute.compute_phase('vision'):
            pytest.fail('symlink lock cannot be trusted')
    assert victim.read_text() == 'keep'
    assert not lifecycle


def test_unverified_native_shutdown_poison_blocks_thread_and_process_lock(policy, lifecycle):
    import fcntl

    with pytest.raises(compute.ComputeShutdownError, match='child exit unknown'):
        with compute.compute_phase('whisper'):
            raise compute.ComputeShutdownError('child exit unknown')
    try:
        with pytest.raises(compute.ComputeShutdownError, match='blocked'):
            with compute.compute_phase('vision'):
                pytest.fail('unreleased native resources must block the next phase')
        lock = os.open(json.loads(policy.read_text())['lock_path'], os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(lock)
    finally:
        os.close(compute._blocked[0])
        compute._blocked = None


def test_actual_lifecycle_waits_for_runner_after_daemon_exits(policy, monkeypatch):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    daemon = compute._Process(100, 1, 100, os.getuid(), 'start', f'{cfg.executable} serve')
    runner = compute._Process(101, 100, 100, os.getuid(), 'start', 'ollama runner')
    late_runner = compute._Process(102, 1, 100, os.getuid(), 'later', 'ollama runner')
    snapshots = iter([{100: daemon, 101: runner}, {101: runner, 102: late_runner}, {102: late_runner}, {}])
    states = iter([(True, 100), (False, None), (False, None), (False, None)])
    calls = []
    monkeypatch.setattr(lifecycle, '_service', lambda: next(states))
    monkeypatch.setattr(lifecycle, '_port_open', lambda: False)
    monkeypatch.setattr(compute, '_processes', lambda: next(snapshots))
    monkeypatch.setattr(compute, '_run', lambda argv, **kw: calls.append(argv) or type('Response', (), {'returncode': 0})())
    try:
        lifecycle.stop()
        assert calls == [['/bin/launchctl', 'bootout', f'gui/{os.getuid()}/com.example.ollama']]
        assert lifecycle._remembered() == []
    finally:
        os.close(lock)


def test_orphaned_runner_from_previous_failed_cleanup_is_still_checked(policy, monkeypatch):
    from dataclasses import replace

    cfg = replace(compute._load_policy(), shutdown_timeout=.01)
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    runner = compute._Process(101, 1, 100, os.getuid(), 'start', 'ollama runner')
    lifecycle._remember([runner])
    monkeypatch.setattr(lifecycle, '_service', lambda: (False, None))
    monkeypatch.setattr(lifecycle, '_port_open', lambda: False)
    monkeypatch.setattr(compute, '_processes', lambda: {101: runner})
    try:
        with pytest.raises(compute.ComputeShutdownError, match='101'):
            lifecycle.stop()
    finally:
        os.close(lock)


def test_lifecycle_will_not_manage_an_unowned_daemon(policy, monkeypatch):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    daemon = compute._Process(100, 1, 100, os.getuid(), 'start', '/another/app serve')
    monkeypatch.setattr(lifecycle, '_service', lambda: (True, 100))
    monkeypatch.setattr(compute, '_processes', lambda: {100: daemon})
    try:
        with pytest.raises(compute.ComputeShutdownError, match='ownership'):
            lifecycle.stop()
    finally:
        os.close(lock)


def test_bootstrap_uses_only_fixed_arguments_after_shutdown(policy, monkeypatch):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    daemon = compute._Process(100, 1, 100, os.getuid(), 'start', f'{cfg.executable} serve')
    calls = []
    monkeypatch.setattr(compute, '_run', lambda argv, **kw: calls.append(argv) or type('Response', (), {'returncode': 0})())
    monkeypatch.setattr(lifecycle, '_service', lambda: (True, 100))
    monkeypatch.setattr(compute, '_processes', lambda: {100: daemon})
    monkeypatch.setattr(lifecycle, '_ready', lambda: True)
    try:
        lifecycle.start()
        assert calls == [['/bin/launchctl', 'bootstrap', f'gui/{os.getuid()}', str(cfg.agent)]]
    finally:
        os.close(lock)


def test_poison_keeps_lock_even_when_writing_ownership_receipt_fails(policy, lifecycle, monkeypatch):
    import fcntl

    def fail_write(*args):
        raise OSError('disk unavailable')

    monkeypatch.setattr(compute.os, 'write', fail_write)
    path = json.loads(policy.read_text())['lock_path']
    with pytest.raises(compute.ComputeShutdownError):
        with compute.compute_phase('whisper'):
            lock = os.open(path, os.O_RDWR)
            raise compute.ComputeShutdownError('child exit unknown')
    try:
        os.fstat(compute._blocked[0])
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert Path(path).stat().st_mode & 0o777 == 0
    finally:
        os.close(lock)
        os.close(compute._blocked[0])
        compute._blocked = None


def test_shutdown_inspection_interruption_is_an_unreleased_compute_failure(policy, monkeypatch):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)

    def interrupt():
        raise KeyboardInterrupt()

    monkeypatch.setattr(lifecycle, '_service', interrupt)
    try:
        with pytest.raises(compute.ComputeShutdownError):
            lifecycle.stop()
    finally:
        os.close(lock)


@pytest.mark.parametrize('status,body,expected', [
    (200, b'{"models": []}', True),
    (200, b'{"unexpected": []}', False),
    (302, b'{"models": []}', False),
    (200, b'broken json', False),
    (200, b'x' * 65537, False),
])
def test_readiness_uses_direct_bounded_http_without_redirects(policy, monkeypatch, status, body, expected):
    cfg = compute._load_policy()
    calls = []

    class Connection:
        def __init__(self, host, port, timeout):
            calls.append((host, port, timeout))

        def request(self, method, path):
            calls.append((method, path))

        def getresponse(self):
            return type('Response', (), {'status': status, 'read': lambda _, cap: body[:cap]})()

        def close(self):
            calls.append('closed')

    monkeypatch.setenv('HTTP_PROXY', 'http://unreachable.example:1234')
    monkeypatch.setattr(compute.http.client, 'HTTPConnection', Connection)
    assert compute._Lifecycle(cfg, -1)._ready() is expected
    assert calls == [('127.0.0.1', 11434, .5), ('GET', '/api/ps'), 'closed']


def test_stale_pid_reuse_does_not_claim_unrelated_process_group(policy, monkeypatch):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    dead = compute._Process(100, 1, 100, os.getuid(), 'old start', 'ollama runner')
    unrelated = compute._Process(100, 1, 100, os.getuid(), 'new start', 'unrelated app')
    lifecycle._remember([dead])
    monkeypatch.setattr(lifecycle, '_service', lambda: (False, None))
    monkeypatch.setattr(lifecycle, '_port_open', lambda: False)
    monkeypatch.setattr(compute, '_processes', lambda: {100: unrelated})
    try:
        lifecycle.stop()
        assert lifecycle._remembered() == []
    finally:
        os.close(lock)


def test_group_writable_policy_is_rejected(policy, lifecycle):
    policy.chmod(0o660)
    with pytest.raises(RuntimeError, match='not writable by others'):
        with compute.compute_phase('vision'):
            pytest.fail('unsafe policy cannot start compute')


def test_failed_bootout_never_claims_shutdown_complete(policy, monkeypatch):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    daemon = compute._Process(100, 1, 100, os.getuid(), 'start', f'{cfg.executable} serve')
    monkeypatch.setattr(lifecycle, '_service', lambda: (True, 100))
    monkeypatch.setattr(compute, '_processes', lambda: {100: daemon})
    monkeypatch.setattr(compute, '_run', lambda *a, **kw: type('Response', (), {'returncode': 5})())
    try:
        with pytest.raises(compute.ComputeShutdownError, match='Failed to shut down'):
            lifecycle.stop()
        assert lifecycle._remembered() == [daemon]
    finally:
        os.close(lock)


def test_release_only_phase_never_bootstraps_ollama(policy, lifecycle):
    with compute.compute_phase('ollama', start_ollama=False) as strict:
        assert strict is True
    assert lifecycle == ['stopped', 'stopped']


def test_nested_release_reuses_existing_translation_ownership(policy, lifecycle):
    with compute.compute_phase('ollama'):
        with compute.compute_phase('ollama', start_ollama=False):
            assert lifecycle == ['stopped', 'ready']
    assert lifecycle == ['stopped', 'ready', 'stopped']


def test_release_only_phase_cannot_hide_a_nested_model_start(policy, lifecycle):
    with compute.compute_phase('ollama', start_ollama=False):
        with pytest.raises(RuntimeError, match='nested'):
            with compute.compute_phase('ollama'):
                pytest.fail('release-only ownership must not permit inference')


@pytest.mark.parametrize('field,value', [
    ('pid', '101'), ('pid', True), ('pid', 0), ('pid', -1),
    ('parent', '1'), ('group', '100'), ('group', 0), ('uid', '501'),
    ('uid', False), ('started', 1), ('started', ''), ('command', []),
    ('command', ''), ('command', 'ollama\x00runner'),
])
def test_invalid_process_receipt_cannot_discard_a_live_orphan(policy, monkeypatch, field, value):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    runner = compute._Process(101, 1, 100, os.getuid(), 'start', 'ollama runner')
    record = vars(runner).copy()
    record[field] = value
    os.write(lock, json.dumps([record]).encode())
    monkeypatch.setattr(lifecycle, '_service', lambda: (False, None))
    monkeypatch.setattr(lifecycle, '_port_open', lambda: False)
    monkeypatch.setattr(compute, '_processes', lambda: {101: runner})
    try:
        with pytest.raises(compute.ComputeShutdownError, match='ownership records'):
            lifecycle.stop()
    finally:
        os.close(lock)


def _read_fifo(path, finished):
    try:
        compute._owned_file(Path(path), 'policy')
    except RuntimeError:
        finished.set()


def test_fifo_policy_is_rejected_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / 'fifo.json'
    os.mkfifo(path, mode=0o600)
    ctx = multiprocessing.get_context('spawn')
    finished = ctx.Event()
    reader = ctx.Process(target=_read_fifo, args=(str(path), finished))
    try:
        reader.start()
        assert finished.wait(2), 'Policy discovery blocked on a FIFO'
        reader.join(2)
        assert reader.exitcode == 0
    finally:
        if reader.is_alive():
            reader.terminate()
            reader.join(2)


def test_child_lease_is_available_only_inside_strict_ownership(policy, lifecycle):
    with pytest.raises(RuntimeError, match='active'):
        compute.compute_lease_fd()
    with compute.compute_phase('vision'):
        fd = compute.compute_lease_fd()
        assert isinstance(fd, int)
        os.fstat(fd)
        with compute.compute_phase('vision'):
            assert compute.compute_lease_fd() == fd
    with pytest.raises(OSError):
        os.fstat(fd)


def test_no_policy_does_not_add_a_child_lease(tmp_path, monkeypatch):
    monkeypatch.delenv('SUBZERO_COMPUTE_CONFIG', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with compute.compute_phase('vision'):
        assert compute.compute_lease_fd() is None


def _parent_with_native_child(config, child_pid_path, child_gate_path):
    import subprocess
    import sys

    os.environ['SUBZERO_COMPUTE_CONFIG'] = config

    class Lifecycle:
        def __init__(self, policy, lock):
            pass

        def stop(self):
            pass

    compute._Lifecycle = Lifecycle
    with compute.compute_phase('vision'):
        fd = compute.compute_lease_fd()
        script = ('import os, pathlib, time; '
                  'pathlib.Path(__import__("sys").argv[1]).write_text(str(os.getpid())); '
                  '\nwhile not pathlib.Path(__import__("sys").argv[2]).exists(): time.sleep(.02)')
        child = subprocess.Popen([sys.executable, '-c', script, child_pid_path, child_gate_path],
                                 pass_fds=(fd,))
        child.wait()


def test_killed_parent_does_not_release_its_live_native_childs_lease(policy, tmp_path):
    import fcntl
    import signal

    ctx = multiprocessing.get_context('spawn')
    pid_path, gate_path = tmp_path / 'child.pid', tmp_path / 'child.exit'
    parent = ctx.Process(target=_parent_with_native_child,
                         args=(str(policy), str(pid_path), str(gate_path)))
    lock = os.open(json.loads(policy.read_text())['lock_path'], os.O_RDWR | os.O_CREAT, 0o600)
    child_pid = None
    try:
        parent.start()
        deadline = time.monotonic() + 3
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pid_path.exists(), 'Native child did not start'
        child_pid = int(pid_path.read_text())
        parent.kill()
        parent.join(2)
        assert parent.exitcode == -signal.SIGKILL
        os.kill(child_pid, 0)
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate_path.touch()
        deadline = time.monotonic() + 3
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline, 'Child exit did not release compute ownership'
                time.sleep(.02)
    finally:
        gate_path.touch()
        if parent.is_alive():
            parent.terminate()
            parent.join(2)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        os.close(lock)


def test_lock_open_is_nonblocking_before_file_type_validation(policy, monkeypatch, lifecycle):
    config = json.loads(policy.read_text())
    path = Path(config['lock_path'])
    os.mkfifo(path, mode=0o600)
    original_open = os.open
    flags_seen = []

    def inspect_open(filename, flags, *args):
        if Path(filename) == path:
            flags_seen.append(flags)
        return original_open(filename, flags, *args)

    monkeypatch.setattr(compute.os, 'open', inspect_open)
    with pytest.raises(RuntimeError, match='regular owned file'):
        with compute.compute_phase('vision'):
            pytest.fail('FIFO lock cannot be trusted')
    assert flags_seen and flags_seen[0] & os.O_NONBLOCK
    assert not lifecycle


def test_process_receipt_rejects_duplicate_pids(policy):
    cfg = compute._load_policy()
    lock = os.open(cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lifecycle = compute._Lifecycle(cfg, lock)
    runner = compute._Process(101, 1, 100, os.getuid(), 'start', 'ollama runner')
    lifecycle._remember([runner, runner])
    try:
        with pytest.raises(RuntimeError, match='ownership records'):
            lifecycle._remembered()
    finally:
        os.close(lock)
