import pytest

from subzero.worker.config import Config


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
