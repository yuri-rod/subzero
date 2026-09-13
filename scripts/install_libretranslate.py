"""Install the pinned native Brazilian Portuguese runtime without loading models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

MANIFEST = Path(__file__).with_name('libretranslate-runtime.json')
PACKAGE = 'translate-en_pb-1_9'
MAX_UNPACKED = 512 * 1024 * 1024
DOWNLOAD_HOSTS = {'files.pythonhosted.org', 'argos-net.com', 'github.com',
                  'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}


def child_environment():
    return {'PATH': os.defpath, 'LANG': 'en_US.UTF-8', 'LC_ALL': 'en_US.UTF-8',
            'UV_NO_CONFIG': '1', 'UV_PYTHON_DOWNLOADS': 'never', 'UV_NO_CACHE': '1'}


def _write_verified(stream, target, digest, size):
    written = 0
    checksum = hashlib.sha256()
    created = False
    try:
        with target.open('xb') as output:
            created = True
            while chunk := stream.read(min(1024 * 1024, size - written + 1)):
                written += len(chunk)
                if written > size:
                    raise ValueError(f'{target.name}: file exceeds pinned size')
                checksum.update(chunk)
                output.write(chunk)
            if written != size or checksum.hexdigest() != digest:
                raise ValueError(f'{target.name}: size or SHA-256 hash differs from pin')
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        if created:
            target.unlink()
        raise


def copy_verified(source, target, digest, size):
    source, target = Path(source), Path(target)
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f'{source.name}: source must be a regular file')
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_NONBLOCK'):
        raise ValueError('Secure source copying requires POSIX file-open protections')
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_size) != (
                before.st_dev, before.st_ino, before.st_mode, before.st_size):
            raise ValueError(f'{source.name}: source changed while opening')
        _write_verified(stream, target, digest, size)
        after = os.fstat(stream.fileno())
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns):
            target.unlink()
            raise ValueError(f'{source.name}: source changed while copying')


def _check_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in DOWNLOAD_HOSTS
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('Download URL is outside the pinned public HTTPS hosts')


class PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_verified(pin, target):
    _check_url(pin['url'])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirects())
    request = urllib.request.Request(pin['url'], headers={'User-Agent': 'ArgosTranslate'})
    with opener.open(request, timeout=60) as response:
        _check_url(response.url)
        _write_verified(response, target, pin['sha256'], pin['size'])


def extract_package(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        paths = set()
        file_paths = set()
        unpacked = 0
        for member in members:
            path = PurePosixPath(member.filename)
            normalized = member.filename.rstrip('/')
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if (not path.parts or path.is_absolute() or '..' in path.parts
                    or member.orig_filename != member.filename
                    or '\\' in member.filename or '\x00' in member.filename
                    or path.parts[0] != PACKAGE or str(path) != normalized
                    or normalized in paths or member.flag_bits & 1
                    or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (kind == stat.S_IFDIR) != member.is_dir() and kind != 0
                    or len(path.parts) == 1 and not member.is_dir()):
                raise ValueError(f'Unsafe or duplicate archive member: {member.filename!r}')
            paths.add(normalized)
            if not member.is_dir():
                file_paths.add(normalized)
            unpacked += member.file_size
            if unpacked > MAX_UNPACKED or len(paths) > 10000:
                raise ValueError('Archive member limits exceeded')
        if any(str(parent) in file_paths for name in paths for parent in PurePosixPath(name).parents):
            raise ValueError('Archive member uses a regular file as a directory')
        try:
            info = archive.getinfo(f'{PACKAGE}/metadata.json')
            if info.file_size > 16384:
                raise ValueError('Package metadata is too large')
            metadata = json.loads(archive.read(info))
        except (KeyError, UnicodeError, json.JSONDecodeError) as err:
            raise ValueError('Package metadata is missing or invalid') from err
        if (not isinstance(metadata, dict) or
                (metadata.get('from_code'), metadata.get('to_code'), metadata.get('package_version'))
                != ('en', 'pb', '1.9')):
            raise ValueError('Package metadata must identify English to Brazilian Portuguese 1.9')
        destination.mkdir(mode=0o700)
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def _load_manifest(path):
    manifest = json.loads(path.read_text('utf-8'))
    if (manifest.get('schema') != 1 or manifest.get('platform') != 'darwin'
            or manifest.get('machine') != 'arm64' or manifest.get('python_version') != '3.13.15'
            or not isinstance(manifest.get('wheels'), list) or len(manifest['wheels']) != 17):
        raise ValueError('Unexpected runtime manifest')
    names = set()
    for pin in [*manifest['wheels'], manifest['archive'], manifest['segmenter']]:
        filename = pin.get('filename', '')
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+', filename) or filename in names
                or not re.fullmatch(r'[0-9a-f]{64}', pin.get('sha256', ''))
                or type(pin.get('size')) is not int or not 0 < pin['size'] <= MAX_UNPACKED):
            raise ValueError('Invalid runtime file pin')
        _check_url(pin['url'])
        names.add(filename)
    for pin in manifest['wheels']:
        if (not pin['filename'].endswith('.whl') or not re.fullmatch(r'[A-Za-z0-9_.-]+', pin['name'])
                or not re.fullmatch(r'[0-9][A-Za-z0-9_.+!-]*', pin['version'])):
            raise ValueError('Invalid dependency pin')
    return manifest


def _run(argv):
    return subprocess.run([str(part) for part in argv], env=child_environment(),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          encoding='utf-8', errors='strict', check=True, timeout=300)


def check_interpreter(python, manifest):
    probe = _run([python, '-I', '-c',
                  'import json,platform,sys; print(json.dumps([platform.python_implementation(), '
                  'platform.python_version(), sys.platform, platform.machine(), platform.mac_ver()[0]]))'])
    identity = json.loads(probe.stdout)
    if (identity[:4] != ['CPython', manifest['python_version'], 'darwin', 'arm64']
            or not identity[4] or int(identity[4].split('.')[0]) < 14):
        raise ValueError('Pinned runtime requires native macOS 14+ ARM64 CPython 3.13.15')
    if not hasattr(os, 'O_NOFOLLOW') or not hasattr(os, 'O_NONBLOCK'):
        raise ValueError('Installer requires POSIX file-open protections')
    return identity


def _create_runtime(runtime):
    for parent in reversed(runtime.parents):
        if parent.is_symlink():
            raise ValueError('Runtime path must not contain symbolic links')
        parent.mkdir(mode=0o700, exist_ok=True)
    runtime.mkdir(mode=0o700)
    return runtime.stat()


def install(runtime, python, uv, manifest_path=MANIFEST, *, model=None, segmenter=None):
    runtime = Path(os.path.abspath(Path(runtime).expanduser()))
    if runtime.exists() or runtime.is_symlink():
        raise ValueError('Runtime already exists; choose a new directory')
    manifest = _load_manifest(Path(manifest_path))
    identity = check_interpreter(python, manifest)
    uv_version = _run([uv, '--version']).stdout.strip()
    created = _create_runtime(runtime)
    try:
        wheels = runtime / 'wheels'
        wheels.mkdir()
        for pin in manifest['wheels']:
            print(f"Fetching {pin['filename']}", flush=True)
            download_verified(pin, wheels / pin['filename'])
        for key, supplied, target in (
                ('archive', model, runtime / manifest['archive']['filename']),
                ('segmenter', segmenter, runtime / 'data/argos-translate/minisbd/en.onnx')):
            target.parent.mkdir(parents=True, exist_ok=True)
            pin = manifest[key]
            if supplied is None:
                download_verified(pin, target)
            else:
                copy_verified(supplied, target, pin['sha256'], pin['size'])
        extract_package(runtime / manifest['archive']['filename'], runtime / 'packages')
        (runtime / 'config').mkdir()
        (runtime / 'cache').mkdir()
        requirements = runtime / 'requirements.txt'
        requirements.write_text(''.join(f"{pin['name']}=={pin['version']} --hash=sha256:{pin['sha256']}\n"
                                        for pin in manifest['wheels']), encoding='utf-8')
        _run([python, '-I', '-m', 'venv', '--without-pip', runtime / '.venv'])
        isolated_python = runtime / '.venv/bin/python'
        _run([uv, 'pip', 'install', '--python', isolated_python, '--no-deps', '--only-binary', ':all:',
              '--require-hashes', '--offline', '--no-index', '--find-links', wheels, '-r', requirements])
        versions = json.loads(_run([isolated_python, '-I', '-c',
                                  'from importlib.metadata import distributions; import json; '
                                  'print(json.dumps({d.metadata["Name"]:d.version for d in distributions()}))']).stdout)
        normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
        expected = {normalize(pin['name']): pin['version'] for pin in manifest['wheels']}
        if {normalize(name): version for name, version in versions.items()} != expected:
            raise ValueError('Installed distributions differ from the pinned dependency set')
        dependency_check = _run([uv, 'pip', 'check', '--python', isolated_python])
        receipt = dict(manifest, interpreter=identity, installer=uv_version,
                       installed_versions=versions, inference_performed=False,
                       dependency_check={'exit_code': dependency_check.returncode,
                                         'stdout': dependency_check.stdout,
                                         'stderr': dependency_check.stderr})
        with (runtime / 'installation.json').open('x', encoding='utf-8') as output:
            json.dump(receipt, output, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        current = runtime.lstat() if runtime.exists() or runtime.is_symlink() else None
        if (current is not None and (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino)
                and not runtime.is_symlink()):
            shutil.rmtree(runtime)
        raise
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path.home() / '.local/share/subzero/libretranslate')
    parser.add_argument('--python', type=Path, required=True, help='Existing native CPython 3.13.15 interpreter')
    parser.add_argument('--uv', type=Path, default=Path(shutil.which('uv') or 'uv'))
    parser.add_argument('--model', type=Path, help='Reuse the pinned .argosmodel file without downloading it')
    parser.add_argument('--segmenter', type=Path, help='Reuse the pinned English en.onnx file')
    args = parser.parse_args()
    try:
        runtime = install(args.runtime, args.python, args.uv, model=args.model, segmenter=args.segmenter)
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as err:
        parser.exit(1, f'Installation failed: {err}\n')
    print(f'Runtime installed at {runtime}; models were not loaded and no translation was run.')


if __name__ == '__main__':
    main()
