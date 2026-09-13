"""Optional machine-wide ownership of Subzero's local compute phases."""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import plistlib
import re
import socket
import stat
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


_gate = threading.RLock()
_owner = threading.local()
_blocked = None


class ComputeShutdownError(RuntimeError):
    """Compute could not be proved stopped; ownership must not pass onward."""

    def __init__(self, message, *, pids=()):
        super().__init__(message)
        self.pids = tuple(pids)


def _endpoint(url):
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if host == 'localhost':
            host = '127.0.0.1'
        if (parsed.scheme != 'http' or not host or not ipaddress.ip_address(host).is_loopback
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/')):
            raise ValueError()
        return host, parsed.port or 80
    except (AttributeError, TypeError, ValueError):
        raise RuntimeError('Compute policy requires a plain loopback Ollama URL') from None


def _owned_file(path, description):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o022):
                raise RuntimeError(f'Compute {description} must be owned by this user and not writable by others')
            raw = handle.read(65537)
        if len(raw) > 65536:
            raise RuntimeError(f'Compute {description} exceeds the size limit')
        return raw
    except OSError as err:
        raise RuntimeError(f'Cannot read compute {description}: {err.strerror}') from None


@dataclass(frozen=True)
class _Policy:
    endpoint: tuple[str, int]
    agent: Path
    label: str
    executable: Path
    lock_path: Path
    startup_timeout: float
    shutdown_timeout: float


def _load_policy():
    custom = os.environ.get('SUBZERO_COMPUTE_CONFIG')
    path = Path(custom) if custom else Path.home() / '.config/subzero/compute.json'
    if not custom and not path.exists() and not path.is_symlink():
        return None
    if not path.is_absolute():
        raise RuntimeError('Compute policy path must be absolute')
    try:
        config = json.loads(_owned_file(path, 'policy'))
        agent = Path(config['ollama_launch_agent'])
        lock = Path(config['lock_path'])
        if not agent.is_absolute() or not lock.is_absolute():
            raise RuntimeError('Compute service and lock paths must be absolute')
        endpoint = _endpoint(config['ollama_url'])
        startup = float(config.get('startup_timeout', 30))
        shutdown = float(config.get('shutdown_timeout', 30))
        if any(not math.isfinite(value) or not 0 < value <= 300 for value in (startup, shutdown)):
            raise RuntimeError('Compute startup and shutdown timeouts must be between 0 and 300 seconds')
        job = plistlib.loads(_owned_file(agent, 'launch agent'))
        args = job.get('ProgramArguments', [])
        if (not isinstance(args, list) or len(args) != 2 or args[1] != 'serve'
                or not isinstance(args[0], str)):
            raise RuntimeError('Compute launch agent must run only an absolute Ollama serve executable')
        executable = Path(args[0])
        if (not executable.is_absolute() or executable.name != 'ollama' or not executable.is_file()
                or not os.access(executable, os.X_OK) or job.get('Program', str(executable)) != str(executable)):
            raise RuntimeError('Compute launch agent must run an installed absolute Ollama serve executable')
        info = executable.stat()
        if info.st_uid not in (0, os.getuid()) or info.st_mode & 0o022:
            raise RuntimeError('Compute Ollama executable has unsafe ownership or permissions')
        if lock.resolve() in (path.resolve(), agent.resolve(), executable.resolve()):
            raise RuntimeError('Compute lock cannot overwrite its policy or service files')
        label = job['Label']
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', label):
            raise RuntimeError('Compute launch agent has an invalid label')
        host = job.get('EnvironmentVariables', {}).get('OLLAMA_HOST', '127.0.0.1:11434')
        if _endpoint(host if '://' in host else f'http://{host}') != endpoint:
            raise RuntimeError('Compute policy URL does not match the managed Ollama service')
        return _Policy(endpoint, agent, label, executable, lock, startup, shutdown)
    except (KeyError, TypeError, ValueError, AttributeError, plistlib.InvalidFileException):
        raise RuntimeError('Invalid compute policy or Ollama launch agent') from None


def _run(argv, timeout=5):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, 'LC_ALL': 'C'})
    except (OSError, subprocess.TimeoutExpired) as err:
        raise RuntimeError(f'Compute lifecycle command failed: {Path(argv[0]).name}') from err


@dataclass(frozen=True)
class _Process:
    pid: int
    parent: int
    group: int
    uid: int
    started: str
    command: str


def _processes():
    response = _run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,uid=,lstart=,command='])
    if response.returncode:
        raise RuntimeError('Cannot inspect compute process ownership')
    processes = {}
    for line in response.stdout.splitlines():
        columns = line.split(None, 9)
        if len(columns) != 10:
            raise RuntimeError('Cannot parse compute process ownership')
        try:
            process = _Process(int(columns[0]), int(columns[1]), int(columns[2]), int(columns[3]),
                               ' '.join(columns[4:9]), columns[9])
        except ValueError:
            raise RuntimeError('Cannot parse compute process identity') from None
        processes[process.pid] = process
    return processes


def _same_process(left, right):
    return (left.pid, left.uid, left.started) == (right.pid, right.uid, right.started)


class _Lifecycle:
    def __init__(self, policy, lock):
        self.policy = policy
        self.lock = lock
        self.target = f'gui/{os.getuid()}/{policy.label}'

    def _service(self):
        response = _run(['/bin/launchctl', 'print', self.target])
        if response.returncode:
            if 'Could not find service' in response.stdout + response.stderr:
                return False, None
            raise RuntimeError('Cannot inspect the managed Ollama launch agent')
        match = re.search(r'^\s*pid = (\d+)\s*$', response.stdout, re.MULTILINE)
        return True, int(match[1]) if match else None

    def _managed(self, pid, processes):
        process = processes.get(pid)
        commands = {f'{self.policy.executable} serve', f'{self.policy.executable.resolve()} serve'}
        if (process is None or process.uid != os.getuid() or process.group != pid
                or process.command not in commands):
            raise RuntimeError('Managed Ollama daemon process ownership could not be verified')
        return process

    def _port_open(self):
        try:
            with socket.create_connection(self.policy.endpoint, timeout=.2):
                return True
        except ConnectionRefusedError:
            return False
        except OSError as err:
            raise RuntimeError('Cannot verify that the Ollama port is closed') from err

    def _remembered(self):
        os.lseek(self.lock, 0, os.SEEK_SET)
        raw = os.read(self.lock, 65537)
        if not raw:
            return []
        if len(raw) > 65536:
            raise RuntimeError('Compute lock contains oversized process ownership records')
        try:
            records = json.loads(raw)
            if not isinstance(records, list):
                raise ComputeShutdownError('An earlier compute shutdown requires independent cleanup verification')
            processes = []
            seen = set()
            for record in records:
                if not isinstance(record, dict) or set(record) != set(_Process.__dataclass_fields__):
                    raise ValueError()
                for field in ('pid', 'parent', 'group', 'uid'):
                    minimum = 1 if field in ('pid', 'group') else 0
                    if type(record[field]) is not int or not minimum <= record[field] <= 2**31 - 1:
                        raise ValueError()
                for field, limit in (('started', 128), ('command', 65536)):
                    value = record[field]
                    if (not isinstance(value, str) or not value.strip() or len(value) > limit
                            or any(char in value for char in ('\x00', '\r', '\n'))):
                        raise ValueError()
                if record['pid'] in seen:
                    raise ValueError()
                seen.add(record['pid'])
                processes.append(_Process(**record))
            return processes
        except (TypeError, ValueError):
            raise RuntimeError('Compute lock contains invalid process ownership records') from None

    def _remember(self, processes):
        # Retain descendants across a failed shutdown, including orphaned runners.
        raw = json.dumps([vars(process) for process in processes]).encode()
        os.lseek(self.lock, 0, os.SEEK_SET)
        os.ftruncate(self.lock, 0)
        os.write(self.lock, raw)
        os.fsync(self.lock)

    def stop(self):
        try:
            self._stop()
        except (RuntimeError, OSError, KeyboardInterrupt, SystemExit) as err:
            if isinstance(err, ComputeShutdownError):
                raise
            raise ComputeShutdownError(str(err)) from err

    def _stop(self):
        deadline = time.monotonic() + self.policy.shutdown_timeout
        loaded, pid = self._service()
        processes = _processes()
        scoped = {p.pid: p for p in self._remembered()
                  if p.pid in processes and _same_process(p, processes[p.pid])}
        if pid is not None:
            scoped[pid] = self._managed(pid, processes)
            descendants = {pid}
            while True:
                children = {p.pid for p in processes.values() if p.parent in descendants} - descendants
                if not children:
                    break
                descendants.update(children)
            for child in descendants:
                process = processes[child]
                if process.uid != os.getuid():
                    raise RuntimeError('Ollama descendant ownership could not be verified')
                scoped[child] = process
        groups = {process.group for process in scoped.values()}
        self._remember(scoped.values())
        if loaded:
            response = _run(['/bin/launchctl', 'bootout', self.target], timeout=self.policy.shutdown_timeout)
            if response.returncode:
                raise RuntimeError('Failed to shut down the managed Ollama launch agent')
        while True:
            loaded, _ = self._service()
            processes = _processes()
            for process in processes.values():
                if process.group in groups and process.uid == os.getuid():
                    scoped[process.pid] = process
            self._remember(scoped.values())
            alive = [p.pid for p in scoped.values() if p.pid in processes and _same_process(p, processes[p.pid])]
            if not loaded and not alive and not self._port_open():
                self._remember([])
                return
            if time.monotonic() >= deadline:
                raise ComputeShutdownError(f'Ollama shutdown was not confirmed; remaining owned PIDs: {alive}', pids=alive)
            time.sleep(.05)

    def _ready(self):
        connection = http.client.HTTPConnection(*self.policy.endpoint, timeout=.5)
        try:
            connection.request('GET', '/api/ps')
            response = connection.getresponse()
            raw = response.read(65537)
            if response.status != 200 or len(raw) > 65536:
                return False
            body = json.loads(raw)
            if not isinstance(body, dict) or not isinstance(body.get('models'), list):
                return False
            if body['models']:
                raise RuntimeError('The managed Ollama service loaded a model before acquiring its compute phase')
            return True
        except (OSError, http.client.HTTPException, ValueError):
            return False
        finally:
            connection.close()

    def start(self):
        deadline = time.monotonic() + self.policy.startup_timeout
        response = _run(['/bin/launchctl', 'bootstrap', f'gui/{os.getuid()}', str(self.policy.agent)],
                        timeout=self.policy.startup_timeout)
        if response.returncode:
            raise RuntimeError('Failed to start the managed Ollama launch agent')
        while True:
            loaded, pid = self._service()
            if loaded and pid is not None:
                processes = _processes()
                try:
                    self._managed(pid, processes)
                except RuntimeError:
                    # launchd can publish a PID before setting its final UID and group.
                    pass
                else:
                    if self._ready():
                        return
            if time.monotonic() >= deadline:
                raise RuntimeError('Managed Ollama did not become ready before the startup timeout')
            time.sleep(.05)


def _open_lock(path, timeout):
    import fcntl

    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError as err:
        raise RuntimeError(f'Cannot open compute lock: {err.strerror}') from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise RuntimeError('Compute lock must be a regular owned file without other writers')
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Another Subzero process still owns the compute phase') from None
                time.sleep(.05)
    except BaseException:
        os.close(fd)
        raise


def _poison(lock, kind, error):
    global _blocked
    _blocked = (lock, str(error))
    if error.pids:
        try:
            processes = _processes()
            records = [vars(processes[pid]) for pid in error.pids if pid in processes]
        except RuntimeError:
            records = {'unverified_phase': kind}
    else:
        records = {'unverified_phase': kind}
    raw = json.dumps(records).encode()
    os.lseek(lock, 0, os.SEEK_SET)
    os.ftruncate(lock, 0)
    os.write(lock, raw)
    os.fsync(lock)


def compute_lease_fd():
    """Return the active lock descriptor to retain with a native child's pass_fds.

    The child shares the flock, so a crashed parent cannot release live compute.
    """
    if _blocked is not None:
        raise ComputeShutdownError('Compute remains blocked after an unverified shutdown')
    active = getattr(_owner, 'phase', None)
    if active is not None:
        return active[3]
    if _load_policy() is not None:
        raise RuntimeError('A native child lease requires an active compute phase')
    return None


@contextmanager
def compute_phase(kind, *, ollama_url=None, start_ollama=True):
    """Yield whether strict ownership is active, releasing services before handoff.

    Callers must wait for native children to exit before leaving their phase.
    An Ollama release-only phase skips startup while retaining shutdown checks.
    """
    if _blocked is not None:
        raise ComputeShutdownError('Compute remains blocked after an unverified shutdown')
    if kind not in ('ollama', 'vision', 'whisper'):
        raise ValueError('Unknown Subzero compute phase')
    if not isinstance(start_ollama, bool) or (not start_ollama and kind != 'ollama'):
        raise ValueError('Only an Ollama release phase can skip service startup')
    active = getattr(_owner, 'phase', None)
    if active:
        if active[0] != kind:
            raise RuntimeError('A nested compute phase cannot switch compute kinds')
        if kind == 'ollama' and start_ollama and not active[2]:
            raise RuntimeError('A nested compute phase cannot start a release-only service')
        if ollama_url is not None and _endpoint(ollama_url) != active[1].endpoint:
            raise RuntimeError('Ollama URL differs from the managed compute policy')
        yield True
        return
    policy = _load_policy()
    if policy is None:
        yield False
        return
    if ollama_url is not None and _endpoint(ollama_url) != policy.endpoint:
        raise RuntimeError('Ollama URL differs from the managed compute policy')
    timeout = policy.startup_timeout + policy.shutdown_timeout
    if not _gate.acquire(timeout=timeout):
        raise RuntimeError('Another Subzero thread still owns the compute phase')
    lock = None
    try:
        if _blocked is not None:
            raise ComputeShutdownError('Compute remains blocked after an unverified shutdown')
        lock = _open_lock(policy.lock_path, timeout)
        lifecycle = _Lifecycle(policy, lock)
        lifecycle.stop()
        try:
            if kind == 'ollama' and start_ollama:
                lifecycle.start()
            _owner.phase = (kind, policy, start_ollama, lock)
            try:
                yield True
            finally:
                _owner.phase = None
        finally:
            if kind == 'ollama':
                lifecycle.stop()
    except ComputeShutdownError as err:
        if lock is not None:
            retained = lock
            lock = None
            try:
                _poison(retained, kind, err)
            except OSError as receipt_error:
                try:
                    os.fchmod(retained, 0)
                except OSError:
                    pass
                raise ComputeShutdownError('Compute remains locked; failed to record unverified shutdown') from receipt_error
        raise
    finally:
        if lock is not None:
            os.close(lock)
        _gate.release()
