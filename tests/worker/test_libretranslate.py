import json
import subprocess
from pathlib import Path

import pytest

from subzero.worker import libretranslate as libre
from subzero.worker.srt import Cue


def test_complete_sentences_share_one_child_and_preserve_original_anchors(tmp_path, monkeypatch):
    requests = []
    client = libre.LibreTranslate(tmp_path)
    monkeypatch.setattr(client, '_run', lambda request: requests.append(request) or {
        'outputs': ['GABE: alpha beta gamma delta', 'SUE: epsilon zeta']})
    cues = [Cue(4, 1.0, 2.0, 'GABE: I want to be'),
            Cue(5, 2.1, 3.0, 'GABE: the first person.'),
            Cue(8, 3.1, 4.0, 'SUE: Another sentence.')]
    before = list(cues)
    lines = client.translate_block(cues, 'pt-BR', 'eng', context={'previous_cues': ['Do not send this']})
    assert requests == [{'action': 'translate', 'texts': [
        'GABE: I want to be the first person.', 'SUE: Another sentence.']}]
    assert len(lines) == 3 and ' '.join(' '.join(lines[:2]).split()) == 'GABE: alpha beta gamma delta'
    assert lines[2] == 'SUE: epsilon zeta'
    assert cues == before


@pytest.mark.parametrize('source,target', [('es', 'pt-BR'), ('en', 'pt-PT'), (None, 'pt-BR'), ('en', 'es')])
def test_no_language_guess_or_generic_portuguese_fallback(tmp_path, monkeypatch, source, target):
    client = libre.LibreTranslate(tmp_path)
    monkeypatch.setattr(client, '_run', lambda req: pytest.fail('Unsupported language reached engine'))
    with pytest.raises(RuntimeError, match='English.*Brazilian Portuguese'):
        client.translate_block([Cue(1, 1, 2, 'A sentence.')], target, source)


@pytest.mark.parametrize('outputs', [[], [''], ['one', 'two'], [4], ['a\x00b'], ['\ud800']])
def test_invalid_output_fails_instead_of_dropping_or_remapping_cues(tmp_path, monkeypatch, outputs):
    client = libre.LibreTranslate(tmp_path)
    monkeypatch.setattr(client, '_run', lambda req: {'outputs': outputs})
    with pytest.raises(RuntimeError, match='output'):
        client.translate_block([Cue(1, 1, 2, 'A sentence.')], 'pt-BR', 'en')


def test_too_few_generated_words_for_anchors_fails(tmp_path, monkeypatch):
    client = libre.LibreTranslate(tmp_path)
    monkeypatch.setattr(client, '_run', lambda req: {'outputs': ['one']})
    with pytest.raises(RuntimeError, match='fewer words'):
        client.translate_block([Cue(1, 1, 2, 'The sentence'), Cue(2, 2, 3, 'continues.')], 'pb', 'en')


def test_cache_identity_pins_engine_models_and_cpu_settings(tmp_path):
    client = libre.LibreTranslate(tmp_path)
    assert client.provider == 'libretranslate'
    assert client.needs_local_compute is False
    assert client.uses_sentence_units is True and client.supports_context is False
    settings = client.cache_settings
    assert settings['package_sha256'] == libre.PACKAGE_SHA256
    assert settings['segmenter_sha256'] == libre.SEGMENTER_SHA256
    assert settings['dependencies']['argos-translate-lt'] == '1.12.1'
    assert settings['options']['device'] == 'cpu'
    assert settings['options']['beam_size'] == 4
    assert settings['source'] == 'en' and settings['target'] == 'pb'


def test_available_only_runs_check_without_loading_engine(tmp_path, monkeypatch):
    client = libre.LibreTranslate(tmp_path)
    calls = []
    monkeypatch.setattr(client, '_run', lambda req: calls.append(req) or {'ready': True})
    client.ensure_available()
    client.release()
    assert calls == [{'action': 'check'}]


def test_isolated_child_uses_fixed_argv_no_shell_and_benchmark_cpu_environment(tmp_path, monkeypatch):
    client = libre.LibreTranslate(tmp_path)
    client.python.parent.mkdir(parents=True)
    client.python.touch()
    captured = {}
    monkeypatch.setenv('DEEPL_API_KEY', 'private-placeholder')
    monkeypatch.setenv('GEMINI_API_KEY', 'private-placeholder')

    class Child:
        returncode = 0
        def __init__(self, argv, **kw):
            captured.update(argv=argv, kw=kw)
            self.out = kw['stdout']
        def communicate(self, payload, timeout):
            captured.update(payload=json.loads(payload), timeout=timeout)
            self.out.write(b'{"ready": true}')
        def poll(self):
            return self.returncode

    monkeypatch.setattr(libre.subprocess, 'Popen', Child)
    client.ensure_available()
    assert captured['argv'][:2] == [str(client.python), '-I']
    assert captured['argv'][2] == str(Path(libre.__file__).resolve())
    assert captured['payload'] == {'action': 'check'}
    assert not captured['kw'].get('shell', False)
    env = captured['kw']['env']
    assert env['ARGOS_DEVICE_TYPE'] == 'cpu'
    assert env['ARGOS_PACKAGES_DIR'] == str(tmp_path / 'packages')
    assert env['ARGOS_BEAM_SIZE'] == '4' and env['OMP_NUM_THREADS'] == '2'
    assert 'DEEPL_API_KEY' not in env and 'GEMINI_API_KEY' not in env


def test_timeout_kills_and_reaps_child_before_returning(tmp_path, monkeypatch):
    client = libre.LibreTranslate(tmp_path, timeout=1)
    client.python.parent.mkdir(parents=True)
    client.python.touch()
    calls = []

    class Child:
        returncode = None
        def __init__(self, *args, **kwargs):
            pass
        def communicate(self, payload, timeout):
            raise subprocess.TimeoutExpired('child', timeout)
        def poll(self):
            return self.returncode
        def kill(self):
            calls.append('kill')
        def wait(self, timeout):
            calls.append('wait')
            self.returncode = -9
            return -9

    monkeypatch.setattr(libre.subprocess, 'Popen', Child)
    with pytest.raises(RuntimeError, match='timed out'):
        client.ensure_available()
    assert calls == ['kill', 'wait']


def test_python_network_calls_are_blocked_before_import_or_translation(monkeypatch):
    from types import SimpleNamespace
    class FakeSocket:
        def connect(self, address):
            pytest.fail('Network reached')
        def connect_ex(self, address):
            pytest.fail('Network reached')
    socket = SimpleNamespace(socket=FakeSocket, create_connection=lambda *args: pytest.fail('Network reached'))
    monkeypatch.setattr(libre, 'socket', socket)
    libre._disable_network()
    with pytest.raises(RuntimeError, match='network'):
        socket.socket().connect(('127.0.0.1', 1))
    with pytest.raises(RuntimeError, match='network'):
        socket.create_connection(('127.0.0.1', 1))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    root = tmp_path / 'runtime'
    installed = root / 'packages' / 'translate-en_pb-1_9'
    installed.mkdir(parents=True)
    contents = {'metadata.json': json.dumps({'from_code': 'en', 'to_code': 'pb', 'package_version': '1.9'}),
                'model/model.bin': 'pinned model'}
    archive = root / 'translate-en_pb-1_9.argosmodel'
    with zipfile.ZipFile(archive, 'w') as output:
        for name, text in contents.items():
            path = installed / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            output.writestr('translate-en_pb-1_9/' + name, text)
    segmenter = root / 'data' / 'argos-translate' / 'minisbd' / 'en.onnx'
    segmenter.parent.mkdir(parents=True)
    segmenter.write_bytes(b'pinned segmenter')
    monkeypatch.setattr(libre, 'PACKAGE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(libre, 'SEGMENTER_SHA256', hashlib.sha256(segmenter.read_bytes()).hexdigest())
    def version(name):
        if name not in libre.DEPENDENCIES:
            raise libre.metadata.PackageNotFoundError(name)
        return libre.DEPENDENCIES[name]
    monkeypatch.setattr(libre.metadata, 'version', version)
    return root


def test_readiness_checks_pinned_files_without_importing_engine(runtime, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith('argostranslate'):
            pytest.fail('Readiness imported the model engine')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    libre._verify_runtime(runtime)


@pytest.mark.parametrize('mutation', ['model', 'extra', 'segmenter', 'fallback', 'package_link'])
def test_changed_models_or_extra_language_never_pass_readiness(runtime, mutation):
    installed = runtime / 'packages' / 'translate-en_pb-1_9'
    if mutation == 'model':
        (installed / 'model' / 'model.bin').write_bytes(b'changed')
    elif mutation == 'extra':
        (installed / 'extra').write_bytes(b'extra')
    elif mutation == 'segmenter':
        (runtime / 'data' / 'argos-translate' / 'minisbd' / 'en.onnx').write_bytes(b'changed')
    elif mutation == 'fallback':
        (runtime / 'packages' / 'translate-en_pt-1_9').mkdir()
    else:
        packages = runtime / 'packages'
        packages.rename(runtime / 'elsewhere')
        try:
            packages.symlink_to(runtime / 'elsewhere', target_is_directory=True)
        except OSError:
            pytest.skip('Symbolic links are unavailable on this test host')
    with pytest.raises(RuntimeError):
        libre._verify_runtime(runtime)


def test_wrong_engine_version_fails_before_translation(runtime, monkeypatch):
    monkeypatch.setattr(libre.metadata, 'version', lambda name: '0.0')
    with pytest.raises(RuntimeError, match='version'):
        libre._verify_runtime(runtime)


def test_cancellation_reaps_child_and_preserves_exception(tmp_path, monkeypatch):
    client = libre.LibreTranslate(tmp_path)
    client.python.parent.mkdir(parents=True)
    client.python.touch()
    calls = []
    class Child:
        returncode = None
        def __init__(self, *args, **kwargs):
            pass
        def communicate(self, payload, timeout):
            raise KeyboardInterrupt()
        def poll(self):
            return self.returncode
        def kill(self):
            calls.append('kill')
        def wait(self, timeout):
            calls.append('wait')
            self.returncode = -9
    monkeypatch.setattr(libre.subprocess, 'Popen', Child)
    with pytest.raises(KeyboardInterrupt):
        client.ensure_available()
    assert calls == ['kill', 'wait']


def test_oversized_output_is_rejected(tmp_path, monkeypatch):
    client = libre.LibreTranslate(tmp_path)
    client.python.parent.mkdir(parents=True)
    client.python.touch()
    class Child:
        returncode = 0
        def __init__(self, *args, **kwargs):
            self.output = kwargs['stdout']
        def communicate(self, payload, timeout):
            self.output.write(b'x' * (libre.MAX_BYTES + 1))
    monkeypatch.setattr(libre.subprocess, 'Popen', Child)
    with pytest.raises(RuntimeError, match='size limit'):
        client.ensure_available()


def test_real_timed_out_cpu_child_is_gone_before_the_adapter_returns(tmp_path, monkeypatch):
    import sys
    client = libre.LibreTranslate(tmp_path, timeout=0.05)
    client.python = Path(sys.executable)
    popen = subprocess.Popen
    children = []
    def sleeping_child(argv, **kwargs):
        child = popen([sys.executable, '-I', '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(libre.subprocess, 'Popen', sleeping_child)
    with pytest.raises(RuntimeError, match='timed out'):
        client.ensure_available()
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdin.closed


def test_failed_child_shutdown_is_reported_as_unverified(tmp_path, monkeypatch):
    client = libre.LibreTranslate(tmp_path, timeout=1)
    client.python.parent.mkdir(parents=True)
    client.python.touch()
    class Child:
        def __init__(self, *args, **kwargs):
            pass
        def communicate(self, payload, timeout):
            raise subprocess.TimeoutExpired('child', timeout)
        def poll(self):
            return None
        def kill(self):
            pass
        def wait(self, timeout):
            raise subprocess.TimeoutExpired('child', timeout)
    monkeypatch.setattr(libre.subprocess, 'Popen', Child)
    with pytest.raises(RuntimeError, match='shutdown could not be verified'):
        client.ensure_available()


def test_runtime_dependency_identity_matches_the_complete_installer_lock(tmp_path):
    lock = Path(__file__).parents[2] / 'scripts' / 'libretranslate-runtime.json'
    expected = {wheel['name']: wheel['version'] for wheel in json.loads(lock.read_text())['wheels']}
    assert len(expected) == 17
    assert libre.LibreTranslate(tmp_path).cache_settings['dependencies'] == expected


@pytest.mark.parametrize('entry', ['getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'sendto'])
def test_dns_and_datagram_calls_are_blocked_without_touching_host_sockets(monkeypatch, entry):
    from types import SimpleNamespace
    def reached(*args, **kwargs):
        pytest.fail('Network entrypoint reached')
    class FakeSocket:
        sendto = staticmethod(reached)
    socket = SimpleNamespace(socket=FakeSocket, create_connection=reached,
                             getaddrinfo=reached, gethostbyname=reached, gethostbyname_ex=reached)
    monkeypatch.setattr(libre, 'socket', socket)
    libre._disable_network()
    call = getattr(socket.socket(), entry) if entry == 'sendto' else getattr(socket, entry)
    with pytest.raises(RuntimeError, match='network'):
        call('example.invalid')
