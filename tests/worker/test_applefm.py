import json

import httpx
import pytest

from subzero.worker import applefm
from subzero.worker.srt import Cue


MODELS = {'object': 'list', 'data': [{'id': 'system', 'object': 'model', 'owned_by': 'Apple'}]}
GUARDRAIL = {'error': {'code': '500', 'type': 'server_error',
                       'message': "The model's safety guardrails were triggered."}}


def adapter(monkeypatch, handler, **kw):
    seen = []
    answer_models = kw.pop('answer_models', True)

    def respond(req):
        seen.append(req)
        if answer_models and req.url.path == '/v1/models':
            return httpx.Response(200, json=MODELS)
        return handler(req)

    client_class = httpx.Client

    def client(**kw):
        return client_class(transport=httpx.MockTransport(respond), **kw)

    monkeypatch.setattr(applefm.httpx, 'Client', client)
    monkeypatch.setattr(applefm.time, 'sleep', lambda _: None)
    return applefm.AppleFM('http://127.0.0.1:1976', **kw), seen


def completion(text):
    return httpx.Response(200, json={'choices': [{'message': {'content': text}}]})


def stream(text):
    lines = [f'data: {json.dumps({"choices": [{"delta": {"content": part}}]})}'
             for part in (text[:len(text) // 2], text[len(text) // 2:])]
    return httpx.Response(200, text='\n\n'.join(lines) + '\n\ndata: [DONE]\n\n')


def test_block_sends_one_request_per_unit_and_returns_one_line_per_cue(monkeypatch):
    replies = iter(['GABE: alfa beta gama delta', 'SUE: épsilon zeta'])
    client, seen = adapter(monkeypatch, lambda req: completion(next(replies)))
    cues = [Cue(4, 1.0, 2.0, 'GABE: I want to be'),
            Cue(5, 2.1, 3.0, 'GABE: the first person.'),
            Cue(8, 3.1, 4.0, 'SUE: Another sentence.')]
    before = list(cues)
    lines = client.translate_block(cues, 'pt-BR', 'en',
                                   context={'previous_cues': ['Earlier line.']})
    assert len(seen) == 2
    payload = json.loads(seen[0].content)
    assert payload['model'] == 'system' and payload['temperature'] == 0
    user = payload['messages'][1]['content']
    assert 'Earlier line.' in user
    assert 'GABE: I want to be\nthe first person.' in user
    assert 'SUE: Another sentence.' in json.loads(seen[1].content)['messages'][1]['content']
    assert len(lines) == 3 and lines[2] == 'SUE: épsilon zeta'
    assert cues == before


def test_streamed_completion_is_accepted(monkeypatch):
    client, _ = adapter(monkeypatch, lambda req: stream('Vá em frente.'))
    lines = client.translate_block([Cue(1, 0, 1, 'Go on.')], 'pt-BR', 'en')
    assert lines == ['Vá em frente.']


def test_refused_unit_falls_back_without_losing_the_block(monkeypatch):
    used = []

    class Libre:
        def translate_block(self, cues, target, source_lang=None, *, context=None):
            used.append((list(cues), target, source_lang))
            return ['Linha reserva.']

    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(500, json=GUARDRAIL)
        return completion('Vá em frente.')

    client, seen = adapter(monkeypatch, handler, fallback=Libre())
    lines = client.translate_block([Cue(1, 0, 1, 'I really need to vote now.'),
                                    Cue(2, 1, 2, 'Go on.')], 'pt-BR', 'en')
    assert lines == ['Linha reserva.', 'Vá em frente.']
    assert len(seen) == 2 and len(used) == 1
    assert [cue.text for cue in used[0][0]] == ['I really need to vote now.']


def test_echoed_unit_retries_once_then_falls_back(monkeypatch):
    used = []

    class Libre:
        def translate_block(self, cues, target, source_lang=None, *, context=None):
            used.append(True)
            return ['Você não pode votar...']

    client, seen = adapter(monkeypatch, lambda req: completion('you cannot vote...'),
                           fallback=Libre())
    lines = client.translate_block([Cue(1, 0, 1, 'you cannot vote...')], 'pt-BR', 'en')
    assert lines == ['Você não pode votar...']
    assert len(seen) == 2 and len(used) == 1


def test_proper_name_echo_is_accepted(monkeypatch):
    client, seen = adapter(monkeypatch, lambda req: completion('Gata!'))
    assert client.translate_block([Cue(1, 0, 1, 'Gata!')], 'pt-BR', 'en') == ['Gata!']
    assert len(seen) == 1


def test_refusal_field_is_treated_as_guardrail(monkeypatch):
    def handler(req):
        return httpx.Response(200, json={'choices': [{'message': {
            'role': 'assistant', 'content': '', 'refusal': 'I cannot help.'}}]})

    client, _ = adapter(monkeypatch, handler)
    with pytest.raises(applefm.AppleFMGuardrail, match='refused'):
        client.translate_block([Cue(1, 0, 1, 'Go.')], 'pt-BR', 'en')


def test_refusal_without_fallback_raises_guardrail(monkeypatch):
    client, seen = adapter(monkeypatch, lambda req: httpx.Response(500, json=GUARDRAIL))
    with pytest.raises(applefm.AppleFMGuardrail, match='guardrails'):
        client.translate_block([Cue(1, 0, 1, 'you cannot vote...')], 'pt-BR', 'en')
    assert len(seen) == 1


def test_transient_errors_retry_then_fail(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(503, json={'error': 'overloaded'})

    client, _ = adapter(monkeypatch, handler)
    with pytest.raises(RuntimeError, match='503'):
        client.translate_block([Cue(1, 0, 1, 'Go.')], 'pt-BR', 'en')
    assert len(calls) == 3


def test_client_errors_are_not_retried(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(400, json={'error': 'bad request'})

    client, _ = adapter(monkeypatch, handler)
    with pytest.raises(RuntimeError, match='rejected.*400'):
        client.translate_block([Cue(1, 0, 1, 'Go.')], 'pt-BR', 'en')
    assert len(calls) == 1


@pytest.mark.parametrize('source,target', [('es', 'pt-BR'), ('en', 'pt-PT'), (None, 'pt-BR'), ('en', 'es')])
def test_no_language_guess_or_generic_portuguese_fallback(monkeypatch, source, target):
    client, seen = adapter(monkeypatch, lambda req: pytest.fail('Unsupported language reached server'))
    with pytest.raises(RuntimeError, match='English.*Brazilian Portuguese'):
        client.translate_block([Cue(1, 1, 2, 'A sentence.')], target, source)
    assert seen == []


@pytest.mark.parametrize('url', ['https://127.0.0.1:1976', 'http://192.168.1.5:1976',
                                 'http://user:pass@127.0.0.1:1976',
                                 'http://127.0.0.1:1976/v1', 'http://127.0.0.1:1976?x=1'])
def test_nonlocal_or_decorated_endpoints_are_rejected(url):
    with pytest.raises(ValueError, match='loopback'):
        applefm.AppleFM(url)


def test_empty_completion_fails_instead_of_dropping_cues(monkeypatch):
    client, _ = adapter(monkeypatch, lambda req: completion('   '))
    with pytest.raises(RuntimeError, match='invalid completion'):
        client.translate_block([Cue(1, 0, 1, 'Go on.')], 'pt-BR', 'en')


def test_invalid_context_fails_before_any_request(monkeypatch):
    client, seen = adapter(monkeypatch, lambda req: pytest.fail('Invalid context reached server'))
    with pytest.raises(RuntimeError, match='context'):
        client.translate_block([Cue(1, 0, 1, 'Go.')], 'pt-BR', 'en', context={'previous_cues': ['a\x00b']})
    assert seen == []


def test_availability_requires_system_model(monkeypatch):
    replies = []

    def handler(req):
        replies.append(req)
        if req.url.path == '/v1/models' and len(replies) == 2:
            return httpx.Response(200, json={'object': 'list', 'data': []})
        return httpx.Response(200, json=MODELS)

    client, _ = adapter(monkeypatch, handler, answer_models=False)
    client.ensure_available()
    with pytest.raises(RuntimeError, match='system model'):
        client.ensure_available()


def test_unreachable_server_names_the_serve_command(monkeypatch):
    def down(req):
        raise httpx.ConnectError('refused')

    client, _ = adapter(monkeypatch, down, answer_models=False)
    with pytest.raises(RuntimeError, match='fm serve'):
        client.ensure_available()


def test_fallback_must_translate_blocks(tmp_path):
    with pytest.raises(ValueError, match='fallback'):
        applefm.AppleFM('http://127.0.0.1:1976', fallback=object())
