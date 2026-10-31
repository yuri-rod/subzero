import importlib.util
import json
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from subzero import compute

posix_only = pytest.mark.skipif(
    os.name != 'posix' or importlib.util.find_spec('fcntl') is None,
    reason='Shared compute lock requires POSIX fcntl',
)


def test_linux_snapshot_uses_available_memory_and_swap_in_bytes(monkeypatch):
    monkeypatch.setattr(compute.sys, 'platform', 'linux')
    read_text = compute.Path.read_text
    def read(path, *args, **kwargs):
        if str(path).replace('\\', '/') == '/proc/meminfo':
            return 'MemTotal: 1000 kB\nMemAvailable: 250 kB\nSwapTotal: 200 kB\nSwapFree: 50 kB\n'
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(compute.Path, 'read_text', read)
    monkeypatch.setattr(compute.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '100\n200\n'))
    result = compute.sample_resources()
    assert {k: result[k] for k in ('memory_total_bytes', 'memory_available_bytes',
                                  'swap_used_bytes', 'swap_total_bytes', 'vram_free_bytes')} == {
        'memory_total_bytes': 1024000, 'memory_available_bytes': 256000,
        'swap_used_bytes': 153600, 'swap_total_bytes': 204800, 'vram_free_bytes': 104857600}
    assert isinstance(result['timestamp'], str)
    json.dumps(result, allow_nan=False)


def test_macos_snapshot_applies_vm_page_size_and_swap_units(monkeypatch):
    monkeypatch.setattr(compute.sys, 'platform', 'darwin')
    responses = {
        ('/usr/bin/vm_stat',): 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 5.\n',
        ('/usr/sbin/sysctl', '-n', 'hw.memsize'): '1073741824\n',
        ('/usr/sbin/sysctl', '-n', 'vm.swapusage'): 'total = 2.00G  used = 512.00M  free = 1.50G (encrypted)\n',
        ('nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'): '',
    }
    monkeypatch.setattr(compute.subprocess, 'run', lambda argv, **kw: subprocess.CompletedProcess(argv, 0, responses[tuple(argv)]))
    result = compute.sample_resources()
    assert result['memory_total_bytes'] == 1073741824
    assert result['memory_available_bytes'] == 573440
    assert result['swap_used_bytes'] == 536870912
    assert result['swap_total_bytes'] == 2147483648
    assert result['vram_free_bytes'] is None


def test_unavailable_resource_commands_produce_unknown_metrics(monkeypatch):
    monkeypatch.setattr(compute.sys, 'platform', 'darwin')
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('unavailable')
    monkeypatch.setattr(compute.subprocess, 'run', unavailable)
    assert all(value is None for key, value in compute.sample_resources().items() if key != 'timestamp')


def test_disabled_pressure_guard_does_not_sample_or_touch_services(monkeypatch):
    def unexpected():
        pytest.fail('disabled guard must not sample resources')
    monkeypatch.setattr(compute, 'sample_resources', unexpected)
    guard = compute.PressureWatchdog()
    guard.check()
    assert guard.status() == {'enabled': False, 'blocked': False, 'reason': '', 'snapshot': None}


@pytest.mark.parametrize('limits,snapshot', [
    ({'min_memory_available_bytes': 100}, {'memory_available_bytes': 99}),
    ({'max_swap_used_bytes': 100}, {'swap_used_bytes': 101}),
    ({'min_vram_free_bytes': 100}, {'vram_free_bytes': 99}),
    ({'min_vram_free_bytes': 100}, {'vram_free_bytes': None}),
])
def test_enabled_thresholds_reject_pressure_and_unknown_metrics(monkeypatch, limits, snapshot):
    monkeypatch.setattr(compute, 'sample_resources', lambda: snapshot)
    guard = compute.PressureWatchdog(**limits, retry_seconds=12)
    with pytest.raises(compute.ComputePressureError) as error:
        guard.check()
    assert error.value.retry_after == 12
    assert guard.status()['blocked']
    assert guard.status()['reason']
    assert compute._blocked is None


def test_periodic_pressure_check_recovers_and_blocks_new_compute(monkeypatch):
    clock = [0.0]
    resources = {'memory_available_bytes': 9}
    monkeypatch.setattr(compute.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(compute, 'sample_resources', lambda: dict(resources))
    guard = compute.PressureWatchdog(min_memory_available_bytes=10, interval_seconds=5)
    compute.install_pressure_watchdog(guard)
    try:
        with pytest.raises(compute.ComputePressureError):
            with compute.compute_phase('vision'):
                pytest.fail('resource pressure must prevent a new phase')
        resources['memory_available_bytes'] = 10
        clock[0] = 4
        assert guard.poll()['blocked']
        clock[0] = 5
        with compute.compute_phase('vision') as strict:
            assert strict is False
        assert not guard.status()['blocked']
        assert compute._blocked is None
    finally:
        compute.install_pressure_watchdog(None)


def test_pressure_never_stops_services_without_policy(monkeypatch):
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': 0})
    monkeypatch.setattr(compute, '_load_policy', lambda: None)
    def unsafe(*args):
        pytest.fail('unmanaged service must never be stopped')
    monkeypatch.setattr(compute._Lifecycle, 'stop', unsafe)
    assert compute.PressureWatchdog(min_memory_available_bytes=1).poll()['blocked']


def test_pressure_does_not_interrupt_an_active_compute_owner(monkeypatch):
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': 0})
    monkeypatch.setattr(compute, '_load_policy', lambda: SimpleNamespace())
    owned = threading.Event()
    release = threading.Event()
    def owner():
        with compute._gate:
            owned.set()
            assert release.wait(3)
    thread = threading.Thread(target=owner)
    thread.start()
    assert owned.wait(1)
    def unsafe(*args):
        pytest.fail('active service must never be stopped')
    monkeypatch.setattr(compute._Lifecycle, 'stop', unsafe)
    try:
        assert compute.PressureWatchdog(min_memory_available_bytes=1).poll()['blocked']
    finally:
        release.set()
        thread.join(3)


@posix_only
def test_pressure_cleanup_keeps_another_process_lease_intact(tmp_path, monkeypatch):
    import fcntl
    lock_path = tmp_path / 'compute.lock'
    lock_path.write_text('native child ownership')
    lock_path.chmod(0o600)
    monkeypatch.setattr(compute, '_load_policy', lambda: SimpleNamespace(lock_path=lock_path))
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': 0})
    with lock_path.open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert compute.PressureWatchdog(min_memory_available_bytes=1).poll()['blocked']
        assert lock_path.read_text() == 'native child ownership'
        assert compute._blocked is None


@posix_only
def test_pressure_cleanup_uses_owned_policy_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / 'compute.lock'
    lock_path.touch(mode=0o600)
    inode = lock_path.stat().st_ino
    policy = SimpleNamespace(lock_path=lock_path, label='test.ollama', shutdown_timeout=.1)
    monkeypatch.setattr(compute, '_load_policy', lambda: policy)
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': 0})
    monkeypatch.setattr(compute._Lifecycle, '_service', lambda self: (False, None))
    monkeypatch.setattr(compute, '_processes', lambda: {})
    monkeypatch.setattr(compute._Lifecycle, '_port_open', lambda self: False)
    assert compute.PressureWatchdog(min_memory_available_bytes=1).poll()['blocked']
    assert lock_path.read_text() == '[]'
    assert lock_path.stat().st_ino == inode
    assert compute._blocked is None


@posix_only
def test_unverified_pressure_cleanup_retains_the_native_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / 'compute.lock'
    policy = SimpleNamespace(lock_path=lock_path, label='test.ollama')
    monkeypatch.setattr(compute, '_load_policy', lambda: policy)
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': 0})
    monkeypatch.setattr(compute, '_blocked', None)
    def failed_shutdown(self):
        raise compute.ComputeShutdownError('Managed process exit is unverified')
    monkeypatch.setattr(compute._Lifecycle, 'stop', failed_shutdown)
    try:
        state = compute.PressureWatchdog(min_memory_available_bytes=1).poll()
        assert state['blocked']
        assert 'unverified' in state['reason']
        assert json.loads(lock_path.read_text()) == {'unverified_phase': 'ollama'}
        with pytest.raises(RuntimeError, match='owns the compute phase'):
            compute._open_lock(lock_path, 0)
    finally:
        if compute._blocked is not None:
            compute.os.close(compute._blocked[0])


@posix_only
def test_pressure_is_rechecked_after_waiting_for_compute_lease(tmp_path, monkeypatch):
    lock_path = tmp_path / 'compute.lock'
    policy = SimpleNamespace(lock_path=lock_path, startup_timeout=1, shutdown_timeout=1)
    monkeypatch.setattr(compute, '_load_policy', lambda: policy)
    clock = [0.0]
    available = [10]
    monkeypatch.setattr(compute.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': available[0]})
    acquire = compute._open_lock
    def wait_for_lease(path, timeout):
        lease = acquire(path, timeout)
        available[0] = 0
        clock[0] = 2
        return lease
    monkeypatch.setattr(compute, '_open_lock', wait_for_lease)
    monkeypatch.setattr(compute, '_Lifecycle', lambda *args: SimpleNamespace(stop=lambda: None))
    guard = compute.PressureWatchdog(min_memory_available_bytes=10, interval_seconds=1)
    compute.install_pressure_watchdog(guard)
    try:
        with pytest.raises(compute.ComputePressureError):
            with compute.compute_phase('vision'):
                pytest.fail('pressure changed while waiting; native work must not start')
        assert compute._blocked is None
        compute.os.close(acquire(lock_path, 0))
    finally:
        compute.install_pressure_watchdog(None)


def test_pressure_keeps_existing_nested_phase_and_release_only_cleanup_running(monkeypatch):
    policy = SimpleNamespace(endpoint=('127.0.0.1', 11434))
    monkeypatch.setattr(compute._owner, 'phase', ('ollama', policy, True, 123), raising=False)
    monkeypatch.setattr(compute, 'sample_resources', lambda: {'memory_available_bytes': 0})
    guard = compute.PressureWatchdog(min_memory_available_bytes=10)
    compute.install_pressure_watchdog(guard)
    try:
        with compute.compute_phase('ollama') as strict:
            assert strict
            assert compute.compute_lease_fd() == 123
        compute._owner.phase = None
        with compute.compute_phase('ollama', start_ollama=False) as strict:
            assert not strict
        assert compute._blocked is None
    finally:
        compute.install_pressure_watchdog(None)


def test_background_watchdog_samples_without_interrupting_running_work(monkeypatch):
    sampled = threading.Event()
    def resources():
        sampled.set()
        return {'memory_available_bytes': 0}
    monkeypatch.setattr(compute, 'sample_resources', resources)
    guard = compute.PressureWatchdog(min_memory_available_bytes=1, interval_seconds=.01)
    guard.start()
    try:
        assert sampled.wait(1)
        assert guard.status()['blocked']
        sampled.clear()
        assert sampled.wait(1)
    finally:
        guard.stop()


@pytest.mark.parametrize('pending_stage', ['sampling', 'cleanup'])
def test_pressure_status_remains_readable_during_slow_resource_work(tmp_path, monkeypatch, pending_stage):
    if pending_stage == 'cleanup' and (os.name != 'posix' or importlib.util.find_spec('fcntl') is None):
        pytest.skip('Compute lock cleanup requires POSIX fcntl')
    entered = threading.Event()
    release = threading.Event()
    observed = threading.Event()
    snapshots = []
    sample = {'memory_available_bytes': 0}
    def wait():
        entered.set()
        release.wait(3)
    def resources():
        if pending_stage == 'sampling':
            wait()
        return sample
    monkeypatch.setattr(compute, 'sample_resources', resources)
    if pending_stage == 'cleanup':
        policy = SimpleNamespace(lock_path=tmp_path / 'compute.lock', label='test.ollama')
        monkeypatch.setattr(compute, '_load_policy', lambda: policy)
        monkeypatch.setattr(compute._Lifecycle, 'stop', lambda self: wait())
    guard = compute.PressureWatchdog(min_memory_available_bytes=1)
    poller = threading.Thread(target=guard.poll)
    def read_status():
        snapshots.append(guard.status())
        observed.set()
    reader = threading.Thread(target=read_status)
    poller.start()
    try:
        assert entered.wait(1)
        reader.start()
        assert observed.wait(.5), 'Health must not wait for sampling or daemon shutdown'
        assert snapshots[0]['enabled']
        if pending_stage == 'cleanup':
            assert snapshots[0]['blocked']
            assert snapshots[0]['snapshot'] == sample
    finally:
        release.set()
        poller.join(3)
        if reader.ident is not None:
            reader.join(3)


@pytest.mark.parametrize('options', [
    {'min_memory_available_bytes': -1}, {'max_swap_used_bytes': True},
    {'min_vram_free_bytes': 0.5}, {'interval_seconds': 0}, {'retry_seconds': float('inf')},
])
def test_invalid_pressure_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        compute.PressureWatchdog(**options)
