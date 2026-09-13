import hashlib
import importlib.util
import os
import struct
import sys
from pathlib import Path

import pytest


def load_preparer(*, unsupported=False):
    if not unsupported and not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_NONBLOCK")):
        pytest.skip("Secure GGUF preparation requires POSIX file-open flags")
    path = Path(__file__).parents[1] / "src" / "subzero" / "hymt2.py"
    spec = importlib.util.spec_from_file_location("prepare_hymt2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("missing", ["O_NOFOLLOW", "O_NONBLOCK"])
def test_unsupported_secure_open_fails_before_creating_files(tmp_path, monkeypatch, missing):
    prep = load_preparer(unsupported=True)
    monkeypatch.delattr(prep.os, missing, raising=False)
    source, stage = tmp_path / "source.gguf", tmp_path / "prepared"

    def unexpected_open(*args, **kwargs):
        raise AssertionError("Unsupported preparation must not open the source")

    monkeypatch.setattr(prep.os, "open", unexpected_open)
    with pytest.raises(ValueError, match="requires POSIX.*O_NOFOLLOW.*O_NONBLOCK"):
        prep.prepare(source, stage)
    assert not source.exists() and not stage.exists()


def test_unsupported_inspection_cli_reports_platform_requirement(tmp_path, monkeypatch, capsys):
    prep = load_preparer(unsupported=True)
    monkeypatch.delattr(prep.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(sys, "argv", ["hymt2", str(tmp_path / "source.gguf")])
    with pytest.raises(SystemExit) as stopped:
        prep.main()
    assert stopped.value.code == 1
    assert "requires POSIX" in capsys.readouterr().err


def gguf_string(text):
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def make_gguf(path, *, eos=3, eos_type=4, bad_token=None, duplicate=False):
    strings = {
        "general.architecture": "hunyuan-dense",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "hunyuan",
    }
    integers = {
        "hunyuan-dense.block_count": 32,
        "hunyuan-dense.embedding_length": 4096,
        "tokenizer.ggml.bos_token_id": 127958,
        "tokenizer.ggml.eot_token_id": 127960,
        "tokenizer.ggml.padding_token_id": 127961,
        "tokenizer.ggml.seperator_token_id": 127962,
    }
    fields = [gguf_string(k) + struct.pack("<I", 8) + gguf_string(v) for k, v in strings.items()]
    fields += [gguf_string(k) + struct.pack("<II", 4, v) for k, v in integers.items()]
    eos_field = gguf_string("tokenizer.ggml.eos_token_id") + struct.pack("<II", eos_type, eos)
    fields.append(eos_field)
    if duplicate:
        fields.append(eos_field)
    special = {3: "$", 127958: "<|startoftext|>", 127960: "<|eos|>",
               127961: "<|pad|>", 127962: "<|extra_0|>", 127967: "<|extra_5|>"}
    if bad_token is not None:
        special[127960] = bad_token
    tokens = b"".join(gguf_string(special.get(i, "t")) for i in range(128167))
    fields.append(gguf_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 128167) + tokens)
    header = b"GGUF" + struct.pack("<IQQ", 3, 354, len(fields)) + b"".join(fields)
    tensors = b"".join(gguf_string(f"tensor{i}") + struct.pack("<IQIQ", 1, 1, 0, i * 4)
                       for i in range(354))
    header += tensors
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + bytes(range(256)) * 8)


def test_prepares_copy_changing_only_eos_field(tmp_path):
    prep = load_preparer()
    source = tmp_path / "original.gguf"
    make_gguf(source)
    original = source.read_bytes()
    stage = tmp_path / "prepared"

    receipt = prep.prepare(source, stage)

    corrected = (stage / "hy-mt2-7b.gguf").read_bytes()
    offset = receipt["eos_offset"]
    assert source.read_bytes() == original
    assert corrected[:offset] == original[:offset]
    assert corrected[offset:offset + 4] == struct.pack("<I", 127960)
    assert corrected[offset + 4:] == original[offset + 4:]
    assert receipt["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["corrected_sha256"] == hashlib.sha256(corrected).hexdigest()
    assert receipt["tensor_sha256"] == hashlib.sha256(original[receipt["tensor_data_offset"]:]).hexdigest()
    assert (stage / "verification.json").is_file()
    modelfile = (stage / "Modelfile").read_text()
    assert 'FROM ./hy-mt2-7b.gguf' in modelfile
    assert '<|startoftext|>{{ .Prompt }}<|extra_0|>' in modelfile
    assert 'PARAMETER stop "<|eos|>"' in modelfile
    assert 'PARAMETER stop "<|extra_5|>"' in modelfile
    assert 'hy_User' not in modelfile


@pytest.mark.parametrize("kwargs", [
    {"eos": 4}, {"eos": 127960}, {"eos_type": 5},
    {"bad_token": "wrong"}, {"duplicate": True},
])
def test_rejects_unexpected_metadata_before_copy(tmp_path, kwargs):
    prep = load_preparer()
    source = tmp_path / "source.gguf"
    make_gguf(source, **kwargs)
    with pytest.raises(ValueError):
        prep.prepare(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_refuses_existing_staging_directory_and_symlink_source(tmp_path):
    prep = load_preparer()
    source = tmp_path / "source.gguf"
    make_gguf(source)
    stage = tmp_path / "prepared"
    stage.mkdir()
    (stage / "keep").write_text("existing content")
    with pytest.raises(FileExistsError):
        prep.prepare(source, stage)
    assert (stage / "keep").read_text() == "existing content"
    link = tmp_path / "source-link.gguf"
    link.symlink_to(source)
    with pytest.raises(OSError):
        prep.prepare(link, tmp_path / "other")


@pytest.mark.parametrize("header", [b"bad", b"GGUF" + struct.pack("<IQQ", 3, 354, 2**63)])
def test_rejects_truncated_and_unbounded_headers(tmp_path, header):
    prep = load_preparer()
    source = tmp_path / "source.gguf"
    source.write_bytes(header)
    with pytest.raises(ValueError):
        prep.prepare(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_detects_source_changes_before_creating_modelfile(tmp_path, monkeypatch):
    prep = load_preparer()
    source = tmp_path / "source.gguf"
    make_gguf(source)
    inspect = prep.inspect

    def mutate_source(stream, **kwargs):
        receipt = inspect(stream, **kwargs)
        with source.open("ab") as changed:
            changed.write(b"changed")
        return receipt

    monkeypatch.setattr(prep, "inspect", mutate_source)
    stage = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Source GGUF changed"):
        prep.prepare(source, stage)
    assert not (stage / "Modelfile").exists()
    assert not (stage / "verification.json").exists()


def test_detects_corrected_tensor_corruption_before_creating_modelfile(tmp_path, monkeypatch):
    prep = load_preparer()
    source = tmp_path / "source.gguf"
    make_gguf(source)
    inspect = prep.inspect

    def corrupt_copy(stream, **kwargs):
        receipt = inspect(stream, **kwargs)
        if kwargs.get("eos") == 127960:
            with Path(stream.name).open("r+b") as changed:
                changed.seek(-1, 2)
                changed.write(b"changed")
        return receipt

    monkeypatch.setattr(prep, "inspect", corrupt_copy)
    stage = tmp_path / "prepared"
    with pytest.raises(ValueError, match="byte verification failed"):
        prep.prepare(source, stage)
    assert not (stage / "Modelfile").exists()
    assert not (stage / "verification.json").exists()
