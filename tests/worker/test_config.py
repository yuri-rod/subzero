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
