import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import urllib.request
import zipfile

import pytest


@pytest.fixture
def installer():
    path = Path(__file__).resolve().parents[1] / 'scripts/install_libretranslate.py'
    if not path.is_file():
        pytest.fail('The scoped runtime installer is not implemented')
    spec = importlib.util.spec_from_file_location('install_libretranslate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_archive(path, extra=None):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('translate-en_pb-1_9/metadata.json', json.dumps({
            'from_code': 'en', 'to_code': 'pb', 'package_version': '1.9'}))
        archive.writestr('translate-en_pb-1_9/model/model.bin', b'model fixture')
        if extra is not None:
            archive.writestr(*extra)
    return path


def test_copy_rejects_wrong_hash_without_leaving_partial_file(installer, tmp_path):
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_NONBLOCK'):
        pytest.skip('Secure copy requires POSIX file-open protections')
    source = tmp_path / 'source'
    source.write_bytes(b'changed model')
    target = tmp_path / 'target'
    with pytest.raises(ValueError, match='hash'):
        installer.copy_verified(source, target, '0' * 64, len(b'changed model'))
    assert not target.exists()
    assert source.read_bytes() == b'changed model'


def test_copy_rejects_symlink_source(installer, tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'known model')
    link = tmp_path / 'link'
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip('Symlink creation is unavailable')
    with pytest.raises(ValueError, match='regular'):
        installer.copy_verified(link, tmp_path / 'target', hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_size)
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('name', ['../outside', '/outside', 'another-package/model.bin',
                                  'translate-en_pb-1_9/../outside', 'translate-en_pb-1_9/model\\escape'])
def test_package_rejects_unsafe_members_before_extraction(installer, tmp_path, name):
    archive = model_archive(tmp_path / 'model.zip', (name, b'bad'))
    destination = tmp_path / 'packages'
    with pytest.raises(ValueError, match='member'):
        installer.extract_package(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / 'outside').exists()


def test_package_rejects_symlink_member_before_extraction(installer, tmp_path):
    member = zipfile.ZipInfo('translate-en_pb-1_9/model/link')
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = model_archive(tmp_path / 'model.zip', (member, b'../../outside'))
    with pytest.raises(ValueError, match='member'):
        installer.extract_package(archive, tmp_path / 'packages')
    assert not (tmp_path / 'packages').exists()


def test_package_rejects_duplicate_destinations(installer, tmp_path):
    with pytest.warns(UserWarning, match='Duplicate'):
        archive = model_archive(tmp_path / 'model.zip', ('translate-en_pb-1_9/model/model.bin', b'other'))
    with pytest.raises(ValueError, match='member'):
        installer.extract_package(archive, tmp_path / 'packages')
    assert not (tmp_path / 'packages').exists()


def test_package_rejects_file_as_parent_before_extraction(installer, tmp_path):
    archive = model_archive(tmp_path / 'model.zip', ('translate-en_pb-1_9/model', b'not a directory'))
    with pytest.raises(ValueError, match='member'):
        installer.extract_package(archive, tmp_path / 'packages')
    assert not (tmp_path / 'packages').exists()


def test_package_extracts_only_pinned_direct_language_metadata(installer, tmp_path):
    archive = model_archive(tmp_path / 'model.zip')
    destination = tmp_path / 'packages'
    installer.extract_package(archive, destination)
    assert (destination / 'translate-en_pb-1_9/model/model.bin').read_bytes() == b'model fixture'
    assert [p.name for p in destination.iterdir()] == ['translate-en_pb-1_9']


def test_existing_runtime_is_not_overwritten(installer, tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    marker = runtime / 'keep'
    marker.write_text('existing runtime')
    with pytest.raises(ValueError, match='already exists'):
        installer.install(runtime, Path('/missing-python'), Path('/missing-uv'), installer.MANIFEST)
    assert marker.read_text() == 'existing runtime'


def test_child_environment_does_not_forward_secrets(installer, monkeypatch):
    monkeypatch.setenv('DEEPL_API_KEY', 'test-secret')
    monkeypatch.setenv('HF_TOKEN', 'other-secret')
    monkeypatch.setenv('PYTHONPATH', '/untrusted')
    env = installer.child_environment()
    assert not {'DEEPL_API_KEY', 'HF_TOKEN', 'PYTHONPATH'}.intersection(env)
    assert not any('secret' in value for value in env.values())


def test_copy_fails_closed_without_no_follow_support(installer, tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.write_bytes(b'known model')
    monkeypatch.delattr(os, 'O_NOFOLLOW', raising=False)
    with pytest.raises(ValueError, match='POSIX'):
        installer.copy_verified(source, tmp_path / 'target', hashlib.sha256(source.read_bytes()).hexdigest(), 11)
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('url', ['http://argos-net.com/model', 'https://user:password@argos-net.com/model',
                                 'https://example.com/model', 'https://argos-net.com:444/model'])
def test_redirects_cannot_downgrade_or_forward_to_unknown_host(installer, url):
    request = urllib.request.Request('https://argos-net.com/model')
    with pytest.raises(ValueError, match='HTTPS'):
        installer.PublicRedirects().redirect_request(request, None, 302, 'Found', {}, url)


def test_wrong_language_is_rejected_before_extraction(installer, tmp_path):
    path = tmp_path / 'wrong-language.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('translate-en_pb-1_9/metadata.json', json.dumps({
            'from_code': 'en', 'to_code': 'pt', 'package_version': '1.9'}))
    with pytest.raises(ValueError, match='Brazilian Portuguese'):
        installer.extract_package(path, tmp_path / 'packages')
    assert not (tmp_path / 'packages').exists()


def test_unsupported_interpreter_does_not_create_runtime(installer, tmp_path, monkeypatch):
    monkeypatch.setattr(installer, '_run', lambda argv: subprocess.CompletedProcess(
        argv, 0, json.dumps(['CPython', '3.14.0', 'darwin', 'arm64', '26.0'])))
    with pytest.raises(ValueError, match='3.13.15'):
        installer.install(tmp_path / 'runtime', Path('/python'), Path('/uv'))
    assert not (tmp_path / 'runtime').exists()


def synthetic_install(installer, tmp_path, monkeypatch, *, failure=None):
    manifest = json.loads(installer.MANIFEST.read_text())
    package = model_archive(tmp_path / 'package.zip').read_bytes()
    manifest['archive'].update(size=len(package), sha256=hashlib.sha256(package).hexdigest())
    manifest['segmenter'].update(size=5, sha256=hashlib.sha256(b'onnx!').hexdigest())
    for pin in manifest['wheels']:
        pin.update(size=5, sha256=hashlib.sha256(b'wheel').hexdigest())
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(installer, 'check_interpreter', lambda *args: ['CPython', '3.13.15', 'darwin', 'arm64', '26.0'])
    def download(pin, target):
        import io
        content = package if pin['filename'].endswith('.argosmodel') else (
            b'onnx!' if pin['filename'] == 'en.onnx' else b'wheel')
        installer._write_verified(io.BytesIO(content), target, pin['sha256'], pin['size'])
    monkeypatch.setattr(installer, 'download_verified', download)
    commands = []
    def run(argv):
        commands.append([str(part) for part in argv])
        if '--version' in argv:
            return subprocess.CompletedProcess(argv, 0, 'uv test')
        if 'venv' in argv:
            (Path(argv[-1]) / 'bin').mkdir(parents=True)
            return subprocess.CompletedProcess(argv, 0, '')
        if 'install' in argv:
            if failure == 'install':
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 0, '')
        versions = {pin['name']: pin['version'] for pin in manifest['wheels']}
        if failure == 'unexpected_dependency':
            versions['torch'] = 'unexpected'
        return subprocess.CompletedProcess(argv, 0, json.dumps(versions))
    monkeypatch.setattr(installer, '_run', run)
    return manifest_path, commands


@pytest.mark.parametrize('failure', ['install', 'unexpected_dependency'])
def test_failed_install_removes_only_its_new_runtime(installer, tmp_path, monkeypatch, failure):
    manifest, _ = synthetic_install(installer, tmp_path, monkeypatch, failure=failure)
    neighbor = tmp_path / 'keep'
    neighbor.write_text('other runtime')
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        installer.install(tmp_path / 'runtime', Path('/python'), Path('/uv'), manifest)
    assert not (tmp_path / 'runtime').exists()
    assert neighbor.read_text() == 'other runtime'


def test_install_is_offline_after_verified_downloads_and_retains_receipt(installer, tmp_path, monkeypatch):
    manifest, commands = synthetic_install(installer, tmp_path, monkeypatch)
    runtime = installer.install(tmp_path / 'runtime', Path('/python'), Path('/uv'), manifest)
    receipt = json.loads((runtime / 'installation.json').read_text())
    assert len(receipt['wheels']) == len(receipt['installed_versions']) == 17
    assert receipt['inference_performed'] is False
    assert list((runtime / 'config').iterdir()) == []
    assert (runtime / 'packages/translate-en_pb-1_9/model/model.bin').read_bytes() == b'model fixture'
    install_command = next(command for command in commands if 'install' in command)
    assert {'--offline', '--no-index', '--no-deps', '--require-hashes'}.issubset(install_command)
    assert not any('import argostranslate' in part for command in commands for part in command)
    assert set(runtime.iterdir()) == {runtime / name for name in (
        'wheels', 'data', 'packages', 'config', 'cache', 'requirements.txt', '.venv',
        'installation.json', 'translate-en_pb-1_9.argosmodel')}
