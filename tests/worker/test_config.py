import subprocess
from types import SimpleNamespace

import pytest

from subzero.worker.config import Config, resolve_deepl_key


@pytest.mark.parametrize('platform,enabled', [('darwin', True), ('linux', False), ('win32', False)])
def test_ocr_default_follows_native_platform(monkeypatch, platform, enabled):
    monkeypatch.setattr('subzero.worker.config.sys.platform', platform)
    env = {'JELLYFIN_URL': 'http://localhost', 'JELLYFIN_API_KEY': 'key', 'BEARER_TOKEN': 'token'}
    assert Config.load(env).ocr_enabled is enabled
    assert Config('http://localhost', 'key', 'token').ocr_enabled is enabled


@pytest.mark.parametrize('value,enabled', [('1', True), ('0', False), ('false', False)])
def test_ocr_setting_can_override_platform(value, enabled):
    env = {'JELLYFIN_URL': 'http://localhost', 'JELLYFIN_API_KEY': 'key', 'BEARER_TOKEN': 'token',
           'OCR_ENABLED': value}
    assert Config.load(env).ocr_enabled is enabled


@pytest.mark.parametrize('model,digest', [('qwen3.5:9b', ''), ('', 'a' * 64)])
def test_ocr_rescue_rejects_partial_model_identity(model, digest):
    with pytest.raises(ValueError, match='both a model and its digest'):
        Config('http://localhost', 'key', 'token', ocr_rescue_model=model,
               ocr_rescue_model_digest=digest)


def test_ocr_rescue_is_opt_in_and_loads_pinned_identity():
    env = {'JELLYFIN_URL': 'http://localhost', 'JELLYFIN_API_KEY': 'key', 'BEARER_TOKEN': 'token'}
    assert not Config.load(env).ocr_rescue_model
    cfg = Config.load(dict(env, OCR_RESCUE_MODEL=' qwen3.5:9b ', OCR_RESCUE_MODEL_DIGEST='a' * 64))
    assert cfg.ocr_rescue_model == 'qwen3.5:9b'
    assert cfg.ocr_rescue_model_digest == 'a' * 64


def test_native_translation_provider_must_be_explicit_and_known():
    env = {'JELLYFIN_URL': 'http://localhost', 'JELLYFIN_API_KEY': 'key', 'BEARER_TOKEN': 'token'}
    assert Config.load(env).translation_provider == 'ollama'
    cfg = Config.load(dict(env, TRANSLATION_PROVIDER='libretranslate', LIBRETRANSLATE_RUNTIME='/local/translation'))
    assert cfg.translation_provider == 'libretranslate'
    assert cfg.libretranslate_runtime == '/local/translation'
    with pytest.raises(ValueError, match='TRANSLATION_PROVIDER'):
        Config.load(dict(env, TRANSLATION_PROVIDER='unknown'))


def test_deepl_configuration_does_not_access_keychain(monkeypatch):
    def forbid_keychain(*args, **kwargs):
        raise AssertionError('Configuration loading must not access credentials')

    monkeypatch.setattr('subzero.worker.config.subprocess.run', forbid_keychain)
    env = {'JELLYFIN_URL': 'http://localhost', 'JELLYFIN_API_KEY': 'key', 'BEARER_TOKEN': 'token',
           'TRANSLATION_PROVIDER': ' DeepL-Free '}
    cfg = Config.load(env)
    assert cfg.translation_provider == 'deepl-free'
    assert cfg.deepl_api_key == ''
    assert Config.load(dict(env, DEEPL_API_KEY=' explicit-free-key:fx ')).deepl_api_key == 'explicit-free-key:fx'


def test_explicit_deepl_key_takes_precedence_on_any_platform(monkeypatch):
    def forbid_keychain(*args, **kwargs):
        raise AssertionError('An explicit key must not access Keychain')

    monkeypatch.setattr('subzero.worker.config.sys.platform', 'linux')
    monkeypatch.setattr('subzero.worker.config.subprocess.run', forbid_keychain)
    cfg = Config('http://localhost', 'key', 'token', deepl_api_key='explicit-free-key:fx')
    assert resolve_deepl_key(cfg) == 'explicit-free-key:fx'


def test_deepl_keychain_reads_fixed_service_without_a_shell(monkeypatch):
    calls = []

    def keychain(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='keychain-free-key:fx\n')

    monkeypatch.setattr('subzero.worker.config.sys.platform', 'darwin')
    monkeypatch.setattr('subzero.worker.config.subprocess.run', keychain)
    assert resolve_deepl_key(Config('http://localhost', 'key', 'token')) == 'keychain-free-key:fx'
    assert calls == [((['/usr/bin/security', 'find-generic-password', '-s',
                       'subzero.deepl.api-free', '-a', 'worker', '-w'],),
                      {'capture_output': True, 'text': True, 'timeout': 10})]


@pytest.mark.parametrize('failure', ['missing', 'empty', 'timeout', 'os-error', 'decode-error'])
def test_deepl_keychain_failure_does_not_expose_output(monkeypatch, failure):
    secret = 'sensitive-keychain-output'

    def keychain(*args, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(args[0], 10, output=secret, stderr=secret)
        if failure == 'os-error':
            raise OSError(secret)
        if failure == 'decode-error':
            raise UnicodeError(secret)
        return SimpleNamespace(returncode=1 if failure == 'missing' else 0,
                               stdout=secret if failure == 'missing' else ' ', stderr=secret)

    monkeypatch.setattr('subzero.worker.config.sys.platform', 'darwin')
    monkeypatch.setattr('subzero.worker.config.subprocess.run', keychain)
    with pytest.raises(ValueError, match='DeepL Free') as error:
        resolve_deepl_key(Config('http://localhost', 'key', 'token'))
    assert secret not in str(error.value)
    assert error.value.__suppress_context__ or error.value.__context__ is None


def test_deepl_missing_key_on_other_platforms_is_explicit(monkeypatch):
    monkeypatch.setattr('subzero.worker.config.sys.platform', 'linux')
    with pytest.raises(ValueError, match='requires DEEPL_API_KEY'):
        resolve_deepl_key(Config('http://localhost', 'key', 'token'))


def test_config_repr_omits_provider_and_service_credentials():
    cfg = Config('http://localhost', 'jellyfin-sensitive', 'bearer-sensitive',
                 opensubtitles_key='opensubtitles-sensitive',
                 opensubtitles_password='password-sensitive', deepl_api_key='deepl-sensitive:fx')
    assert 'sensitive' not in repr(cfg)
