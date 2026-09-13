"""DeepL Free translation with an episode-sized character budget."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from contextlib import contextmanager
import threading
import time

import httpx

from ..translate import SPEAKER, reflow_translation, sentence_units

ENDPOINT = 'https://api-free.deepl.com'
FREE_CHARACTER_LIMIT = 500000
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
OPTIONS = {'split_sentences': 'nonewlines', 'preserve_formatting': True,
           'formality': 'prefer_less', 'model_type': 'prefer_quality_optimized',
           'show_billed_characters': True}
LOG = logging.getLogger(__name__)


class DeepLPause(RuntimeError):
    pause_queue = True


class DeepLQuotaExceeded(DeepLPause):

    def __init__(self, required, remaining):
        self.required = required
        self.remaining = remaining
        super().__init__(f'DeepL Free episode requires {required} characters; '
                         f'{remaining} remain. Queue paused before further translation.')


class DeepLRequestUncertain(DeepLPause):
    pass


class DeepLFree:
    provider = 'deepl-free'
    model = 'deepl-free/en:pt-BR'
    needs_local_compute = False
    uses_sentence_units = True
    supports_context = True
    retry_invalid_output = False

    def __init__(self, api_key, timeout=60, *, usage_path=None):
        if not isinstance(api_key, str) or not re.fullmatch(r'[A-Za-z0-9_-]+:fx', api_key):
            raise ValueError('DeepL requires a valid Free API key')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('DeepL timeout must be positive and finite')
        self._api_key = api_key
        self.timeout = timeout
        self.usage_path = Path(usage_path).expanduser().absolute() if usage_path is not None else None
        self._account = hashlib.sha256(api_key.encode('ascii')).hexdigest()
        self._lock = threading.RLock()
        self._usage_floor = 0
        self._episode_remaining = None
        self._languages_verified = False
        self.quota_snapshot = None

    @property
    def cache_settings(self):
        return {'provider': self.provider, 'model': self.model, 'endpoint': ENDPOINT,
                'source': 'EN', 'target': 'PT-BR', 'options': dict(OPTIONS),
                'input_format': 'complete-unit-newline-joined-v1',
                'context_format': 'previous-and-current-dialogue-v1'}

    @staticmethod
    def _units(cues):
        units = list(sentence_units(cues))
        texts = []
        for unit in units:
            speaker = SPEAKER.match(unit[0].text)
            parts = [unit[0].text]
            for cue in unit[1:]:
                repeated = SPEAKER.match(cue.text)
                parts.append(cue.text[repeated.end():] if speaker and repeated
                             and speaker.group(1) == repeated.group(1) else cue.text)
            text = '\n'.join(parts)
            if not text.strip() or '\x00' in text:
                raise RuntimeError('DeepL source is empty or contains a null character')
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                raise RuntimeError('DeepL source contains invalid Unicode') from None
            texts.append(text)
        return units, texts

    def count_episode(self, cues):
        return sum(len(text) for text in self._units(cues)[1])

    def _verify_languages(self):
        if self._languages_verified:
            return
        languages = self._request('GET', '/v3/languages?resource=translate_text')
        if not isinstance(languages, list):
            raise RuntimeError('DeepL returned an invalid language list')
        english = any(isinstance(lang, dict) and lang.get('lang') == 'en'
                      and lang.get('usable_as_source') is True and lang.get('status') == 'stable'
                      for lang in languages)
        brazilian = any(isinstance(lang, dict) and lang.get('lang') == 'pt-BR'
                        and lang.get('usable_as_target') is True and lang.get('status') == 'stable'
                        for lang in languages)
        if not english or not brazilian:
            raise RuntimeError('DeepL does not report stable English to Brazilian Portuguese support')
        self._languages_verified = True

    def ensure_available(self):
        with self._lock, self._quota_journal():
            self._verify_languages()
            self._check_usage(0)

    def check_quota(self, required):
        if type(required) is not int or required < 0:
            raise ValueError('DeepL episode character count must be a nonnegative integer')
        with self._lock, self._quota_journal():
            self._episode_remaining = None
            self._verify_languages()
            snapshot = self._check_usage(required)
            self._episode_remaining = required
            return snapshot

    def _check_usage(self, required):
        usage = self._request('GET', '/v2/usage')
        if (not isinstance(usage, dict)
                or any(type(usage.get(key)) is not int or usage[key] < 0
                       for key in ('character_count', 'character_limit'))):
            raise RuntimeError('DeepL returned invalid usage counters; translation remains blocked')
        self._usage_floor = max(self._usage_floor, usage['character_count'])
        self._save_usage()
        limit = min(FREE_CHARACTER_LIMIT, usage['character_limit'])
        remaining = max(0, limit - self._usage_floor)
        self.quota_snapshot = {'character_count': self._usage_floor, 'character_limit': limit,
                               'remaining': remaining, 'required': required}
        if required > remaining:
            raise DeepLQuotaExceeded(required, remaining)
        return dict(self.quota_snapshot)

    def release(self):
        with self._lock:
            self._episode_remaining = None

    def translate_block(self, cues, target_lang, source_lang=None, *, context=None):
        if ((source_lang or '').lower() not in {'en', 'eng', 'english'}
                or target_lang.lower().replace('_', '-') not in {'pt-br', 'pb', 'pob'}):
            raise RuntimeError('DeepL requires English source and Brazilian Portuguese target')
        if not cues:
            return []
        units, texts = self._units(cues)
        required = sum(map(len, texts))
        payload = {'text': texts, 'source_lang': 'EN', 'target_lang': 'PT-BR', **OPTIONS}
        previous = context.get('previous_cues', []) if isinstance(context, dict) else []
        if (not isinstance(previous, list)
                or any(not isinstance(text, str) or '\x00' in text for text in previous)):
            raise RuntimeError('DeepL context must contain only subtitle dialogue')
        payload['context'] = '\n'.join(previous[-32:] + texts)[-12000:]
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        except UnicodeEncodeError:
            raise RuntimeError('DeepL context contains invalid Unicode') from None
        if len(encoded) > MAX_REQUEST_BYTES or len(texts) > 50:
            raise RuntimeError('DeepL source block exceeds the request size limit')
        with self._lock, self._quota_journal():
            if self._episode_remaining is None or required > self._episode_remaining:
                raise RuntimeError('DeepL requires full episode quota admission before translation')
            response = self._request('POST', '/v2/translate', encoded, billable=required)
            self._episode_remaining -= required
        translations = response.get('translations') if isinstance(response, dict) else None
        if (not isinstance(translations, list) or len(translations) != len(units)
                or any(not isinstance(row, dict) or not isinstance(row.get('text'), str)
                       or not row['text'].strip() or '\x00' in row['text'] for row in translations)):
            raise RuntimeError('DeepL output does not match the source units')
        for text, row in zip(texts, translations):
            if type(row.get('billed_characters')) is not int or row['billed_characters'] != len(text):
                raise DeepLRequestUncertain('DeepL billing count differs from the submitted source')
            if row.get('detected_source_language') != 'EN':
                raise RuntimeError('DeepL output has an unexpected source language')
            try:
                row['text'].encode('utf-8')
            except UnicodeEncodeError:
                raise RuntimeError('DeepL output contains invalid Unicode') from None
        return [line for unit, row in zip(units, translations)
                for line in reflow_translation(unit, row['text'])]

    @contextmanager
    def _quota_journal(self):
        if self.usage_path is None:
            yield
            return
        lock_fd = None
        locked = False
        try:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock_path = self.usage_path.with_name(self.usage_path.name + '.lock')
            flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
            if lock_path.is_symlink():
                raise DeepLPause('DeepL quota journal lock must not be a symbolic link')
            lock_fd = os.open(lock_path, flags, 0o600)
            opened = os.fstat(lock_fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise DeepLPause('DeepL quota journal lock must be a regular private file')
            if os.name == 'nt':
                import msvcrt
                if opened.st_size == 0:
                    os.write(lock_fd, b'0')
                os.lseek(lock_fd, 0, os.SEEK_SET)
                msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            self._read_usage()
            yield
        except OSError:
            raise DeepLPause('DeepL quota journal could not be safely locked or written') from None
        finally:
            if lock_fd is not None:
                if locked:
                    if os.name == 'nt':
                        os.lseek(lock_fd, 0, os.SEEK_SET)
                        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def _read_usage(self):
        try:
            before = self.usage_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 4096:
            raise DeepLPause('DeepL quota journal must be a regular private file')
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
        with os.fdopen(os.open(self.usage_path, flags), 'rb') as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino)
                    != (opened.st_dev, opened.st_ino)):
                raise DeepLPause('DeepL quota journal changed while opening')
            raw = stream.read(4097)
        try:
            saved = json.loads(raw)
        except (ValueError, UnicodeError):
            raise DeepLPause('DeepL quota journal is invalid; usage reconciliation is required') from None
        if (not isinstance(saved, dict) or saved.get('version') != 1
                or saved.get('account') != self._account
                or type(saved.get('character_count')) is not int or saved['character_count'] < 0):
            raise DeepLPause('DeepL quota journal does not match this account or has invalid counters')
        self._usage_floor = max(self._usage_floor, saved['character_count'])

    def _save_usage(self):
        if self.usage_path is None:
            return
        if self.usage_path.is_symlink():
            raise DeepLPause('DeepL quota journal must not be a symbolic link')
        saved = {'version': 1, 'account': self._account, 'character_count': self._usage_floor,
                 'updated_at': time.time(), 'period_reset': 'manual-confirmation-required'}
        fd, temp_name = tempfile.mkstemp(prefix='.deepl-quota-', dir=self.usage_path.parent)
        tmp = Path(temp_name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(saved, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.usage_path)
            if os.name != 'nt':
                parent = os.open(self.usage_path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
        finally:
            tmp.unlink(missing_ok=True)

    def _request(self, method, path, payload=None, *, billable=0):
        for attempt in range(3):
            if billable:
                self._check_usage(self._episode_remaining)
                # Usage can lag by minutes; reserve every possibly billed attempt before sending.
                self._usage_floor += billable
                self._save_usage()
            try:
                with httpx.Client(timeout=self.timeout, trust_env=False, follow_redirects=False) as client:
                    with client.stream(method, ENDPOINT + path, content=payload, headers={
                            'Authorization': 'DeepL-Auth-Key ' + self._api_key,
                            'Content-Type': 'application/json', 'Accept': 'application/json'}) as response:
                        status = response.status_code
                        trace = response.headers.get('X-Trace-ID', '')
                        if re.fullmatch(r'[A-Za-z0-9._:-]{1,128}', trace) and self._api_key not in trace:
                            LOG.info('DeepL %s %s status=%s trace=%s', method, path, status, trace)
                        raw = bytearray()
                        if status == 200:
                            for chunk in response.iter_bytes():
                                raw.extend(chunk)
                                if len(raw) > MAX_RESPONSE_BYTES:
                                    raise RuntimeError('DeepL output exceeds the size limit')
            except (httpx.HTTPError, OSError):
                if billable:
                    raise DeepLRequestUncertain(
                        'DeepL request outcome is uncertain; automatic replay is blocked') from None
                raise RuntimeError('DeepL account check failed; translation remains blocked') from None
            if status == 200:
                try:
                    return json.loads(raw)
                except (ValueError, UnicodeError):
                    raise RuntimeError('DeepL returned invalid output') from None
            if status == 456:
                raise DeepLQuotaExceeded(self._episode_remaining or 0, 0)
            if status != 429 and not 500 <= status <= 599:
                raise RuntimeError(f'DeepL rejected the request (HTTP {status})')
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise RuntimeError(f'DeepL request failed after three attempts (HTTP {status})')


class FallbackTranslator:
    """Uses a primary provider up to its quota limit, then falls back to a secondary provider."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._current = primary
        self._lock = threading.RLock()

    @property
    def provider(self):
        with self._lock:
            return self._current.provider

    @property
    def model(self):
        with self._lock:
            return self._current.model

    @property
    def needs_local_compute(self):
        with self._lock:
            return getattr(self._current, 'needs_local_compute', False)

    @property
    def uses_sentence_units(self):
        with self._lock:
            return getattr(self._current, 'uses_sentence_units', True)

    @property
    def supports_context(self):
        with self._lock:
            return getattr(self._current, 'supports_context', False)

    @property
    def cache_settings(self):
        with self._lock:
            return self._current.cache_settings

    @property
    def url(self):
        with self._lock:
            return getattr(self._current, 'url', None)

    def count_episode(self, cues):
        if hasattr(self.primary, 'count_episode'):
            return self.primary.count_episode(cues)
        from ..srt import dump
        return len(dump(cues))

    def ensure_available(self):
        with self._lock:
            self.fallback.ensure_available()
            try:
                self.primary.ensure_available()
            except DeepLQuotaExceeded as err:
                LOG.info('Primary provider quota exhausted on startup (%d remaining); routing to %s',
                         err.remaining, self.fallback.provider)
                self._current = self.fallback
            except (DeepLPause, RuntimeError) as err:
                LOG.warning('Primary provider %s unavailable (%s); routing to %s',
                            getattr(self.primary, 'provider', 'primary'), err, self.fallback.provider)
                self._current = self.fallback

    def check_quota(self, required):
        with self._lock:
            try:
                snapshot = self.primary.check_quota(required)
                self._current = self.primary
                return snapshot
            except DeepLQuotaExceeded as err:
                LOG.info('Primary quota exceeded (%d needed, %d remaining); falling back to %s',
                         err.required, err.remaining, self.fallback.provider)
                self._current = self.fallback
                if hasattr(self.fallback, 'check_quota'):
                    return self.fallback.check_quota(required)
                return {'provider': self.fallback.provider, 'fallback': True,
                        'required': required, 'primary_remaining': err.remaining}

    def translate_block(self, cues, target_lang, source_lang=None, *, context=None):
        with self._lock:
            active = self._current
        return active.translate_block(cues, target_lang, source_lang=source_lang, context=context)

    def release(self):
        with self._lock:
            if hasattr(self.primary, 'release'):
                self.primary.release()
            if hasattr(self.fallback, 'release'):
                self.fallback.release()
            self._current = self.primary

