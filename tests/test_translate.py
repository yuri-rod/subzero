import pytest
from subzero.convert import Cue
from subzero.translate import OllamaClient, chunks, translate_cues

def test_chunks_partitioning():
    items = list(range(10))
    res = list(chunks(items, 3))
    assert len(res) == 4
    assert res[0] == [0, 1, 2]

def test_parse_numbered_lines():
    resp = """1. Ola mundo
2. Tudo bem com voce?
3. Ate logo!"""
    lines = OllamaClient._parse_lines(resp)
    assert len(lines) == 3
    assert lines[0] == "Ola mundo"
    assert lines[1] == "Tudo bem com voce?"
