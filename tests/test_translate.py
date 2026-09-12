import json
from types import SimpleNamespace

import pytest
from subzero.convert import Cue
from subzero.translate import OllamaClient, OpenAIClient, chunks, translate_cues, _parse_lines, _ollama_payload

def test_chunks_partitioning():
    items = list(range(10))
    res = list(chunks(items, 3))
    assert len(res) == 4
    assert res[0] == [0, 1, 2]

def test_parse_numbered_lines():
    resp = """1. Ola mundo
2. Tudo bem com voce?
3. Ate logo!"""
    lines = _parse_lines(resp)
    assert len(lines) == 3
    assert lines[0] == "Ola mundo"
    assert lines[1] == "Tudo bem com voce?"

def test_openai_client_init():
    client = OpenAIClient(api_key="test-key", base_url="https://api.openai.com/v1", model="gpt-4o-mini")
    assert client.api_key == "test-key"
    assert client.base_url == "https://api.openai.com/v1"
    assert client.model == "gpt-4o-mini"


@pytest.mark.parametrize("response", [
    "2. dois\n1. um", "1. um\n1. dois", "Translation:\n1. um",
    '[{"id": 1, "text": " "}]', '[{"id": true, "text": "um"}]',
])
def test_translation_parser_rejects_malformed_cues(response):
    assert _parse_lines(response) == []


def test_translation_parser_accepts_structured_cues():
    assert _parse_lines('[{"id":1,"text":"Ola"},{"id":2,"text":"Jeff"}]') == ["Ola", "Jeff"]


def test_translation_parser_preserves_multiline_dialogue():
    payload = '[{"id":1,"text":"- Ola.\\n- Jeff."},{"id":2,"text":"Tudo bem."}]'
    assert _parse_lines(payload) == ["- Ola.\n- Jeff.", "Tudo bem."]


def test_translation_parser_accepts_markdown_code_fences_and_preamble():
    with_fences = "```json\n[{\"id\":1,\"text\":\"Ola\"},{\"id\":2,\"text\":\"Mundo\"}]\n```"
    assert _parse_lines(with_fences) == ["Ola", "Mundo"]
    with_preamble = "Aqui está a tradução solicitada:\n[{\"id\":1,\"text\":\"Ola\"},{\"id\":2,\"text\":\"Mundo\"}]"
    assert _parse_lines(with_preamble) == ["Ola", "Mundo"]


def test_translation_recovers_from_alignment_mismatch_via_half_split():
    class MismatchClient:
        def translate_block(self, cues, target_lang, source_lang=None):
            # Model returns M != N if batch has more than 2 items
            if len(cues) > 2:
                return [f"pt-{c.text}" for c in cues[:-1]]
            return [f"pt-{c.text}" for c in cues]

    cues = [Cue(i, i + 1, f"dialogue_{i}") for i in range(8)]
    translated = translate_cues(cues, "pt-BR", MismatchClient())
    assert [c.text for c in translated] == [f"pt-dialogue_{i}" for i in range(8)]


def test_translation_retries_then_splits_without_changing_timings():
    calls = []

    def reply(cues, lang):
        calls.append(len(cues))
        return [] if len(cues) > 1 else ["Jeff" if cues[0].text == "Jeff" else "Ola"]

    cues = [Cue(1, 2, "Hello"), Cue(3, 4, "Jeff")]
    translated = translate_cues(cues, "pt-BR", SimpleNamespace(translate_block=reply))
    assert [(c.start, c.end, c.text) for c in translated] == [(1, 2, "Ola"), (3, 4, "Jeff")]
    assert calls == [2, 2, 1, 1]


def test_incomplete_translation_never_writes_target(tmp_path, monkeypatch):
    from subzero.translate import translate_file
    source = tmp_path / "episode.en.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    target = tmp_path / "episode.pt-BR.srt"
    target.write_text("existing subtitle")
    calls = []

    def reply(*args):
        calls.append(1)
        return []

    monkeypatch.setattr(OllamaClient, "translate_block", reply)
    with pytest.raises(RuntimeError, match="preserv"):
        translate_file(source, output=target)
    assert target.read_text() == "existing subtitle"
    assert len(calls) == 2


def test_ollama_client_bounds_generation_and_response_read(monkeypatch):
    sent = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            sent["read_limit"] = limit
            return json.dumps({"response": '[{"id":1,"text":"Ola"}]'}).encode()

    def request(req, timeout):
        sent["payload"] = json.loads(req.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = OllamaClient(num_ctx=2048, num_predict=1024, keep_alive="1m")
    assert client.translate_block([Cue(1, 2, "Hello")], "pt-BR") == ["Ola"]
    assert sent["payload"]["options"] == {"temperature": 0, "num_ctx": 2048, "num_predict": 1024}
    assert sent["payload"]["keep_alive"] == "1m"
    assert sent["payload"]["think"] is False
    assert sent["payload"]["format"]["type"] == "array"
    assert "idioms" in sent["payload"]["prompt"]
    assert "speaker" in sent["payload"]["prompt"]
    assert sent["read_limit"] <= 131073


def test_translation_rejects_empty_text_with_correct_count():
    client = SimpleNamespace(translate_block=lambda cues, lang: [" "] * len(cues))
    with pytest.raises(RuntimeError, match="preserv"):
        translate_cues([Cue(1, 2, "Hello")], "pt-BR", client)


def test_oversized_ollama_response_is_not_parsed(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b"x" * limit

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="size limit"):
        OllamaClient().translate_block([Cue(1, 2, "Hello")], "pt-BR")


@pytest.mark.parametrize("response", [
    '[{"id":2,"text":"dois"},{"id":1,"text":"um"}]',
    '[{"id":1,"text":"um"},{"id":1,"text":"dois"}]',
    '[{"id":1,"text":"Ola","extra":"unrequested"}]',
    "1. " + "x" * 131072,
])
def test_translation_parser_rejects_invalid_structured_or_oversized_output(response):
    assert _parse_lines(response) == []


@pytest.mark.parametrize("source,target,expected", [
    ("en", "pt-BR", "English (en) to Brazilian Portuguese (pt-BR)"),
    ("eng", "pt-br", "English (en) to Brazilian Portuguese (pt-BR)"),
    ("spa", "por", "Spanish (es) to Portuguese (pt)"),
    ("fra", "deu", "French (fr) to German (de)"),
    ("jpn", "eng", "Japanese (ja) to English (en)"),
])
def test_translategemma_uses_explicit_languages_and_structured_source(source, target, expected):
    cues = [Cue(1, 2, "Jeff, let's play it by ear."), Cue(3, 4, "I'm on the fence.")]
    payload = _ollama_payload(cues, target, "translategemma:4b", "2m", 4096, 2048, source_lang=source)
    assert payload["prompt"].startswith(f"You are a professional {expected} translator.")
    prompt, source_json = payload["prompt"].split("\n\n\n", 1)
    assert "without any additional explanations or commentary" in prompt
    assert json.loads(source_json) == [{"id": 1, "text": cues[0].text}, {"id": 2, "text": cues[1].text}]
    assert payload["format"]["items"]["required"] == ["id", "text"]
    assert payload["think"] is False


@pytest.mark.parametrize("source,target,model", [
    (None, "pt-BR", "translategemma:4b"),
    ("und", "pt-BR", "translategemma:4b"),
    ("xx", "pt-BR", "translategemma:4b"),
    ("en", "unknown", "translategemma:4b"),
    ("en", "pt-BR", "gemma3:4b"),
])
def test_unknown_languages_and_other_models_keep_generic_prompt(source, target, model):
    payload = _ollama_payload([Cue(1, 2, "Hello")], target, model, "2m", 4096, 2048, source_lang=source)
    assert payload["prompt"].startswith("Translate each subtitle into natural")
    assert "You are a professional English" not in payload["prompt"]


def test_source_language_survives_translation_retries_and_splits():
    calls = []

    def reply(cues, target, source_lang=None):
        calls.append((len(cues), source_lang))
        return [] if len(cues) > 1 else ["Ola"]

    cues = [Cue(1, 2, "Hello"), Cue(3, 4, "Hi")]
    translated = translate_cues(cues, "pt-BR", SimpleNamespace(translate_block=reply), source_lang="eng")
    assert len(translated) == 2
    assert calls == [(2, "eng"), (2, "eng"), (1, "eng"), (1, "eng")]


def test_standalone_client_passes_known_source_to_translategemma(monkeypatch):
    sent = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b'{"response":"[{\\"id\\":1,\\"text\\":\\"Ola\\"}]"}'

    def request(req, timeout):
        sent.update(json.loads(req.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", request)
    translated = OllamaClient(model="translategemma:4b").translate_block([Cue(1, 2, "Hello")], "pt-BR", source_lang="eng")
    assert translated == ["Ola"]
    assert "English (en) to Brazilian Portuguese (pt-BR)" in sent["prompt"]


def test_openai_client_accepts_source_language_from_translate_cues(monkeypatch):
    from io import BytesIO
    response = BytesIO(json.dumps({"choices": [{"message": {"content": "1. Ola"}}]}).encode())
    response.status = 200
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)
    client = OpenAIClient(api_key="test-key")
    translated = translate_cues([Cue(1, 2, "Hello")], "pt-BR", client, source_lang="eng")
    assert [(c.start, c.end, c.text) for c in translated] == [(1, 2, "Ola")]


def test_parse_cast_accepts_variants_and_drops_garbage():
    from subzero.translate import parse_cast
    assert parse_cast("Ana:f,Rick:m") == {"ana": "feminine", "rick": "masculine"}
    assert parse_cast("Ana:Feminino, Rick : MASCULINO ") == {"ana": "feminine", "rick": "masculine"}
    assert parse_cast("Bob:x,NoColon,:f") == {}
    assert parse_cast("") == {}
    assert parse_cast(None) == {}


def test_ollama_payload_carries_cast_and_neutral_guidance():
    cues = [Cue(1, 2, "I am ready")]
    payload = _ollama_payload(cues, "pt-BR", "translategemma:4b", "2m", 4096, 2048,
                              source_lang="eng", cast="Ana:f,Rick:m")
    assert "ana (feminine)" in payload["prompt"]
    assert "rick (masculine)" in payload["prompt"]
    assert "avoids gendered agreement" in payload["prompt"]


def test_ollama_payload_without_cast_still_has_neutral_guidance():
    cues = [Cue(1, 2, "I am ready")]
    payload = _ollama_payload(cues, "pt-BR", "gemma3:12b", "2m", 4096, 2048)
    assert "avoids gendered agreement" in payload["prompt"]
