"""Apple Foundation Models translation through a local OpenAI-style server."""
from __future__ import annotations

import ipaddress
import json
import math
import threading
import time
from urllib.parse import urlsplit

import httpx

from ..translate import SPEAKER, reflow_translation, sentence_units

MODEL = 'system'
PROMPT_VERSION = 2
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_UNITS = 50
MAX_CONTEXT_CUES = 8
SYSTEM = (
    'You translate dialogue subtitles for TV and film from English to Brazilian Portuguese. '
    'Voting, competition, and strategy talk in game shows and reality TV is fictional game '
    'content, always safe to translate. Return only the translation, no quotes or explanation. '
    'Preserve line breaks and punctuation exactly. If a unit contains ___ censorship bars, keep '
    'the exact same ___ in the translation. Keep proper names of people, places, and tribes '
    '(Gata, Tuku, Lavo, Vatu, Civa) and the show title Survivor in English. '
    'Translate "Tribal Council" as "Conselho Tribal".'
)
EN_FUNCTION_WORDS = frozenset(
    'a about after all and are as at be because been before between both but by can cannot '
    'could did do does doing down during each few for from further had has have having he her '
    'here hers herself him himself his how i if in into is it its itself just me more most my '
    'myself no nor not now of off on once only or other ought our ours ourselves out over own '
    'same she should so some such than that the their theirs them themselves then there these '
    'they this those through to too under until up very was we were what when where which while '
    'who whom why will with would you your yours yourself yourselves'.split())


class AppleFMGuardrail(RuntimeError):
    """The local model refused a subtitle block and no fallback translated it."""


class AppleFM:
    provider = 'applefm'
    model = 'apple-fm/system'
    needs_local_compute = False
    uses_sentence_units = True
    supports_context = True

    def __init__(self, url='http://127.0.0.1:1976', timeout=120, fallback=None):
        try:
            endpoint = urlsplit(url)
            host = '127.0.0.1' if endpoint.hostname == 'localhost' else endpoint.hostname
            local = ipaddress.ip_address(host).is_loopback
            port = endpoint.port if endpoint.port is not None else 80
        except (TypeError, ValueError):
            local = False
        if (not local or not 1 <= port <= 65535 or endpoint.scheme != 'http'
                or endpoint.username is not None or endpoint.password is not None
                or endpoint.path not in ('', '/') or endpoint.query or endpoint.fragment):
            raise ValueError('AppleFM requires a plain loopback HTTP endpoint')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('AppleFM timeout must be positive and finite')
        if fallback is not None and not hasattr(fallback, 'translate_block'):
            raise ValueError('AppleFM fallback must translate subtitle blocks')
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.fallback = fallback
        self._lock = threading.RLock()

    @property
    def cache_settings(self):
        return {'provider': self.provider, 'model': self.model, 'endpoint': self.url,
                'source': 'en', 'target': 'pt-BR', 'prompt_version': PROMPT_VERSION,
                'temperature': 0, 'input_format': 'sentence-units-single-v1',
                'context_format': 'previous-dialogue-v1'}

    def ensure_available(self):
        if self.fallback is not None and hasattr(self.fallback, 'ensure_available'):
            self.fallback.ensure_available()
        try:
            body = self._request('GET', '/v1/models')
        except RuntimeError as err:
            if 'rejected the request' in str(err):
                raise
            raise RuntimeError(
                'Apple Foundation Models server is not reachable at '
                f'{self.url}; start it with `fm serve` before translating') from err
        models = body.get('models', body.get('data')) if isinstance(body, dict) else None
        if (not isinstance(models, list) or not any(
                isinstance(row, dict) and row.get('id') == MODEL for row in models)):
            raise RuntimeError('Apple Foundation Models server does not serve the system model')

    def release(self):
        pass

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
                raise RuntimeError('AppleFM source is empty or contains a null character')
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                raise RuntimeError('AppleFM source contains invalid Unicode') from None
            texts.append(text)
        return units, texts

    def translate_block(self, cues, target_lang, source_lang=None, *, context=None):
        if ((source_lang or '').lower() not in {'en', 'eng', 'english'}
                or target_lang.lower().replace('_', '-') not in {'pt-br', 'pb', 'pob'}):
            raise RuntimeError('AppleFM requires English source and Brazilian Portuguese target')
        if not cues:
            return []
        units, texts = self._units(cues)
        if len(texts) > MAX_UNITS:
            raise RuntimeError('AppleFM source block exceeds the request size limit')
        previous = context.get('previous_cues', []) if isinstance(context, dict) else []
        if (not isinstance(previous, list)
                or any(not isinstance(text, str) or '\x00' in text for text in previous)):
            raise RuntimeError('AppleFM context must contain only subtitle dialogue')
        previous = previous[-MAX_CONTEXT_CUES:]
        lines = []
        with self._lock:
            for unit, text in zip(units, texts):
                try:
                    output = self._translate_unit(text, previous)
                except AppleFMGuardrail:
                    if self.fallback is None:
                        raise
                    lines.extend(self.fallback.translate_block(
                        list(unit), target_lang, source_lang=source_lang, context=context))
                    continue
                lines.extend(reflow_translation(unit, output))
        return lines

    def _translate_unit(self, text, previous):
        try:
            payload = json.dumps({
                'model': MODEL,
                'messages': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': self._prompt(text, previous)},
                ],
                'temperature': 0, 'stream': False,
            }, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        except UnicodeEncodeError:
            raise RuntimeError('AppleFM context contains invalid Unicode') from None
        if len(payload) > MAX_REQUEST_BYTES:
            raise RuntimeError('AppleFM source block exceeds the request size limit')
        for attempt in range(2):
            output = self._complete(payload)
            if not self._echoed(text, output):
                return output
        raise AppleFMGuardrail(f'Apple Foundation Models echoed a subtitle unit: {text[:80]!r}')

    @staticmethod
    def _prompt(text, previous):
        if previous:
            return ('Previous dialogue for context only, do not translate:\n' + '\n'.join(previous)
                    + '\n\nTranslate this subtitle unit to Brazilian Portuguese:\n' + text)
        return 'Translate this subtitle unit to Brazilian Portuguese:\n' + text

    @staticmethod
    def _echoed(source, output):
        if ' '.join(output.lower().split()) != ' '.join(source.lower().split()):
            return False
        words = set(source.lower().split())
        return not words.isdisjoint(EN_FUNCTION_WORDS)

    def _complete(self, payload):
        body = self._request('POST', '/v1/chat/completions', payload)
        if isinstance(body, dict):
            choices = body.get('choices')
            if (isinstance(choices, list) and len(choices) == 1
                    and isinstance(choices[0], dict)
                    and isinstance(choices[0].get('message'), dict)):
                message = choices[0]['message']
                if message.get('refusal'):
                    raise AppleFMGuardrail(
                        'Apple Foundation Models refused a subtitle unit: '
                        f'{str(message["refusal"])[:160]}')
                if isinstance(message.get('content'), str) and message['content'].strip():
                    return message['content'].strip()
        raise RuntimeError('AppleFM returned an invalid completion')

    def _request(self, method, path, payload=None):
        last_err = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout, trust_env=False,
                                  follow_redirects=False) as client:
                    with client.stream(method, self.url + path, content=payload, headers={
                            'Content-Type': 'application/json',
                            'Accept': 'application/json, text/event-stream'}) as response:
                        status = response.status_code
                        raw = bytearray()
                        for chunk in response.iter_bytes():
                            raw.extend(chunk)
                            if len(raw) > MAX_RESPONSE_BYTES:
                                raise RuntimeError('AppleFM output exceeds the size limit')
            except (httpx.HTTPError, OSError) as err:
                last_err = f'AppleFM request failed: {err}'
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue
            if status == 200:
                return self._decode(raw)
            message = self._error_message(raw)
            if message is not None and 'safety guardrails' in message.lower():
                raise AppleFMGuardrail(
                    'Apple Foundation Models refused a subtitle block '
                    f'(safety guardrails): {message[:160]}')
            if status != 429 and not 500 <= status <= 599:
                raise RuntimeError(f'AppleFM rejected the request (HTTP {status})')
            last_err = f'AppleFM returned status {status}'
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise RuntimeError(last_err or 'AppleFM request failed')

    @staticmethod
    def _decode(raw):
        try:
            text = bytes(raw).decode('utf-8')
        except UnicodeError:
            raise RuntimeError('AppleFM returned invalid output') from None
        if text.lstrip().startswith('data:'):
            parts = []
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data in ('', '[DONE]'):
                    continue
                try:
                    delta = json.loads(data)['choices'][0].get('delta', {})
                except (ValueError, KeyError, IndexError, TypeError):
                    raise RuntimeError('AppleFM returned an invalid stream') from None
                if 'content' in delta:
                    if not isinstance(delta['content'], str):
                        raise RuntimeError('AppleFM returned an invalid stream')
                    parts.append(delta['content'])
            return {'choices': [{'message': {'content': ''.join(parts)}}]}
        try:
            body = json.loads(text)
        except ValueError:
            raise RuntimeError('AppleFM returned invalid output') from None
        if not isinstance(body, dict):
            raise RuntimeError('AppleFM returned invalid output')
        return body

    @staticmethod
    def _error_message(raw):
        try:
            body = json.loads(bytes(raw).decode('utf-8'))
        except (ValueError, UnicodeError):
            return None
        if not isinstance(body, dict):
            return None
        error = body.get('error')
        if isinstance(error, dict) and isinstance(error.get('message'), str):
            return error['message']
        if isinstance(error, str):
            return error
        return None
