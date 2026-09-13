"""The tested LibreTranslate engine in an isolated, offline CPU process."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile

PACKAGE_SHA256 = '1d1cd5e9540c6b38c258bed002a42d3b311b8a189acb74feaa311ef30d175c5b'
SEGMENTER_SHA256 = '6fa9f3a3b201687bd43e329b2b0736789efc97b37883b5b759b31988dca9f353'
DEPENDENCIES = {
    'argos-translate-lt': '1.12.1', 'ctranslate2': '4.8.2', 'minisbd': '0.9.5',
    'onnxruntime': '1.30.0', 'sentencepiece': '0.2.2', 'numpy': '2.5.3', 'sacremoses': '0.1.1',
    'PyYAML': '6.0.3', 'click': '8.5.0', 'cloudpickle': '3.1.2', 'filelock': '3.32.6',
    'flatbuffers': '25.12.19', 'joblib': '1.6.0', 'packaging': '26.3',
    'protobuf': '7.36.1', 'regex': '2026.9.10', 'tqdm': '4.70.1',
}
OPTIONS = {'device': 'cpu', 'model_provider': 'OPENNMT', 'inter_threads': 1,
           'intra_threads': 2, 'batch_size': 32, 'beam_size': 4,
           'compute_type': 'auto', 'chunk_type': 'DEFAULT', 'num_hypotheses': 1}
MAX_BYTES = 1024 * 1024


def _environment(root):
    env = {key: os.environ[key] for key in (
        'PATH', 'HOME', 'USERPROFILE', 'SYSTEMROOT', 'WINDIR', 'TMPDIR', 'TEMP', 'TMP',
        'LANG', 'LC_ALL', 'LC_CTYPE') if key in os.environ}
    env.update({
        'XDG_DATA_HOME': str(root / 'data'), 'XDG_CONFIG_HOME': str(root / 'config'),
        'XDG_CACHE_HOME': str(root / 'cache'), 'ARGOS_PACKAGES_DIR': str(root / 'packages'),
        'ARGOS_DEVICE_TYPE': 'cpu', 'ARGOS_MODEL_PROVIDER': 'OPENNMT', 'ARGOS_DEBUG': '0',
        'ARGOS_INTER_THREADS': '1', 'ARGOS_INTRA_THREADS': '2', 'ARGOS_BATCH_SIZE': '32',
        'ARGOS_BEAM_SIZE': '4', 'ARGOS_COMPUTE_TYPE': 'auto', 'ARGOS_CHUNK_TYPE': 'DEFAULT',
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'VECLIB_MAXIMUM_THREADS': '2',
    })
    return env


class LibreTranslate:
    provider = 'libretranslate'
    model = 'argos-translate-lt:1.12.1/en-pb:1.9'
    needs_local_compute = False
    uses_sentence_units = True
    supports_context = False

    def __init__(self, runtime_dir, timeout=600):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('LibreTranslate timeout must be positive and finite')
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.python = self.runtime_dir / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        self.timeout = timeout
        self._lock = threading.RLock()

    @property
    def cache_settings(self):
        return {'provider': self.provider, 'model': self.model,
                'dependencies': dict(DEPENDENCIES), 'package_sha256': PACKAGE_SHA256,
                'segmenter_sha256': SEGMENTER_SHA256, 'options': dict(OPTIONS),
                'source': 'en', 'target': 'pb', 'input_format': 'complete-unit-whitespace-joined-v1'}

    def ensure_available(self):
        with self._lock:
            if self._run({'action': 'check'}).get('ready') is not True:
                raise RuntimeError('LibreTranslate runtime verification failed')

    def release(self):
        with self._lock:
            pass

    def translate_block(self, cues, target_lang, source_lang=None, *, context=None):
        from ..translate import SPEAKER, reflow_translation, sentence_units
        if ((source_lang or '').lower() not in {'en', 'eng', 'english'}
                or target_lang.lower().replace('_', '-') not in {'pt-br', 'pb', 'pob', 'pt', 'por'}):
            raise RuntimeError('LibreTranslate requires English source and Brazilian Portuguese target')
        if not cues:
            return []
        units = list(sentence_units(cues))
        texts = []
        for unit in units:
            speaker = SPEAKER.match(unit[0].text)
            parts = [unit[0].text]
            for cue in unit[1:]:
                repeated = SPEAKER.match(cue.text)
                parts.append(cue.text[repeated.end():] if speaker and repeated
                             and speaker.group(1) == repeated.group(1) else cue.text)
            texts.append(' '.join('\n'.join(parts).split()))
        if any(not text or '\x00' in text for text in texts):
            raise RuntimeError('LibreTranslate source is empty or contains a null character')
        with self._lock:
            outputs = self._run({'action': 'translate', 'texts': texts}).get('outputs')
        if (not isinstance(outputs, list) or len(outputs) != len(units)
                or any(not isinstance(text, str) or not text.strip() or '\x00' in text for text in outputs)):
            raise RuntimeError('LibreTranslate output does not match the source units')
        try:
            encoded_size = sum(len(text.encode('utf-8')) for text in outputs)
        except UnicodeEncodeError:
            raise RuntimeError('LibreTranslate output contains invalid Unicode') from None
        if encoded_size > MAX_BYTES:
            raise RuntimeError('LibreTranslate output exceeds the size limit')
        return [line for unit, output in zip(units, outputs) for line in reflow_translation(unit, output)]

    def _run(self, request):
        payload = json.dumps(request, ensure_ascii=True).encode('utf-8')
        if len(payload) > MAX_BYTES:
            raise RuntimeError('LibreTranslate source block exceeds the size limit')
        if not self.python.is_file():
            raise RuntimeError('LibreTranslate runtime is not installed at the configured path')
        argv = [str(self.python), '-I', str(Path(__file__).resolve()), str(self.runtime_dir)]
        with tempfile.TemporaryFile() as output:
            try:
                proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=output,
                                        stderr=subprocess.DEVNULL, env=_environment(self.runtime_dir))
            except OSError:
                raise RuntimeError('LibreTranslate runtime could not be started') from None
            try:
                proc.communicate(payload, timeout=self.timeout)
            except BaseException as err:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('LibreTranslate child shutdown could not be verified') from None
                if isinstance(err, subprocess.TimeoutExpired):
                    raise RuntimeError('LibreTranslate CPU request timed out; child was stopped') from None
                raise
            finally:
                if getattr(proc, 'stdin', None) is not None:
                    proc.stdin.close()
            output.seek(0)
            raw = output.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise RuntimeError('LibreTranslate output exceeds the size limit')
        try:
            response = json.loads(raw)
        except (ValueError, UnicodeError):
            raise RuntimeError('LibreTranslate returned invalid output') from None
        if not isinstance(response, dict):
            raise RuntimeError('LibreTranslate returned invalid output')
        if proc.returncode != 0 or response.get('error'):
            reason = response.get('error', 'child failed')
            if reason not in {'runtime_verification', 'invalid_request', 'engine_failure'}:
                reason = 'child failed'
            raise RuntimeError(f'LibreTranslate failed ({reason}); verify the isolated runtime')
        return response


def _hash_stream(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError('Runtime model must be a regular file')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError('Runtime model changed while opening')
        digest = _hash_stream(stream)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError('Runtime model changed while checking')
    return digest


def _verify_runtime(root):
    for name, version in DEPENDENCIES.items():
        if metadata.version(name) != version:
            raise RuntimeError('Translation dependency version differs from the tested runtime')
    try:
        metadata.version('argostranslate')
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError('Conflicting Argos distributions are installed')
    archive_path = root / 'translate-en_pb-1_9.argosmodel'
    if _hash_file(archive_path) != PACKAGE_SHA256:
        raise RuntimeError('Translation package digest differs from the tested package')
    packages = root / 'packages'
    installed = packages / 'translate-en_pb-1_9'
    if packages.is_symlink() or set(packages.iterdir()) != {installed} or installed.is_symlink():
        raise RuntimeError('Runtime must contain only the direct Brazilian Portuguese package')
    expected = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != installed.name:
                raise RuntimeError('Translation package contains an invalid path')
            if member.is_dir():
                continue
            destination = packages / relative
            if destination.is_symlink() or any((packages / Path(*relative.parts[:n])).is_symlink()
                                               for n in range(1, len(relative.parts))):
                raise RuntimeError('Translation package contains a symbolic link')
            expected.add(destination)
            with archive.open(member) as stream:
                if _hash_stream(stream) != _hash_file(destination):
                    raise RuntimeError('Installed translation package differs from its archive')
    found = {path for path in installed.rglob('*') if path.is_file() or path.is_symlink()}
    if found != expected:
        raise RuntimeError('Installed translation package has unexpected files')
    package_info = json.loads((installed / 'metadata.json').read_text('utf-8'))
    if (package_info.get('from_code'), package_info.get('to_code'), package_info.get('package_version')) != ('en', 'pb', '1.9'):
        raise RuntimeError('Installed translation package is not English to Brazilian Portuguese')
    segmenter = root / 'data' / 'argos-translate' / 'minisbd' / 'en.onnx'
    if _hash_file(segmenter) != SEGMENTER_SHA256:
        raise RuntimeError('English sentence segmenter differs from the tested model')


def _disable_network():
    def unavailable(*args, **kwargs):
        raise RuntimeError('LibreTranslate network access is disabled')
    socket.socket.connect = unavailable
    socket.socket.connect_ex = unavailable
    socket.socket.sendto = unavailable
    socket.create_connection = unavailable
    socket.getaddrinfo = unavailable
    socket.gethostbyname = unavailable
    socket.gethostbyname_ex = unavailable


def _translate_cpu(root, texts):
    from argostranslate import package, settings, translate
    if (settings.device != 'cpu' or settings.model_provider != settings.ModelProvider.OPENNMT
            or settings.package_dirs != [root / 'packages']
            or settings.chunk_type != settings.ChunkType.ARGOSTRANSLATE
            or any(getattr(settings, name) != OPTIONS[name] for name in (
                'inter_threads', 'intra_threads', 'batch_size', 'beam_size', 'compute_type'))):
        raise RuntimeError('Engine settings differ from the tested CPU configuration')
    installed = package.get_installed_packages()
    if len(installed) != 1 or (installed[0].from_code, installed[0].to_code) != ('en', 'pb'):
        raise RuntimeError('Only the direct Brazilian Portuguese package is supported')
    languages = translate.get_installed_languages()
    if {lang.code for lang in languages} != {'en', 'pb'}:
        raise RuntimeError('Unexpected translation language or pivot route')
    english = next(lang for lang in languages if lang.code == 'en')
    brazilian = next(lang for lang in languages if lang.code == 'pb')
    translator = english.get_translation(brazilian)
    if not isinstance(translator, translate.CachedTranslation):
        raise RuntimeError('Unexpected translation engine')
    direct = translator.underlying
    if not isinstance(direct, translate.PackageTranslation) or direct.pkg.to_code != 'pb':
        raise RuntimeError('Translation route is not the direct Brazilian Portuguese package')
    detector = direct.sentencizer.lazy_detector()
    if detector.session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('Sentence segmenter is not using CPU')
    outputs = [translator.translate(text) for text in texts]
    if direct.translator.device != 'cpu' or direct.translator.compute_type != 'int8_float32':
        raise RuntimeError('Translation did not use the tested CPU precision')
    return outputs


def _child(root):
    _disable_network()
    raw = sys.stdin.buffer.read(MAX_BYTES + 1)
    try:
        request = json.loads(raw)
    except (ValueError, UnicodeError):
        return {'error': 'invalid_request'}
    if len(raw) > MAX_BYTES or not isinstance(request, dict):
        return {'error': 'invalid_request'}
    try:
        _verify_runtime(root)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, metadata.PackageNotFoundError):
        return {'error': 'runtime_verification'}
    if request == {'action': 'check'}:
        return {'ready': True}
    texts = request.get('texts')
    if (request.get('action') != 'translate' or not isinstance(texts, list) or not texts
            or any(not isinstance(text, str) or not text.strip() or '\x00' in text for text in texts)):
        return {'error': 'invalid_request'}
    try:
        return {'outputs': _translate_cpu(root, texts)}
    except Exception:
        # Third-party errors can echo subtitle text; keep child diagnostics bounded.
        return {'error': 'engine_failure'}


if __name__ == '__main__':
    response = _child(Path(sys.argv[1]).resolve())
    sys.stdout.write(json.dumps(response, ensure_ascii=True))
    raise SystemExit(1 if response.get('error') else 0)
