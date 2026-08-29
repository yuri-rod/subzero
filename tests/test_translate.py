import pytest
from subzero.convert import Cue
from subzero.translate import OllamaClient, OpenAIClient, chunks, translate_cues, _parse_lines

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
