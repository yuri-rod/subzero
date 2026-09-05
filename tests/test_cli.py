import runpy
import sys

import pytest

from subzero.cli import build_parser


def test_max_line_must_be_positive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fix", "demo.srt", "--max-line", "0"])


def test_settle_must_be_non_negative():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["watch", "demo.srt", "--settle", "-1"])


def test_python_m_subzero_supports_version(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["python -m subzero", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("subzero", run_name="__main__", alter_sys=True)
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("subzero ")
def test_verify_sync_emits_json_and_distinct_status(monkeypatch, tmp_path, capsys):
    from subzero.cli import main
    from subzero import reference
    from subzero.timing import Report
    subtitle = tmp_path/'movie.srt'
    subtitle.write_text('subtitle')
    monkeypatch.setattr(reference,'build_reference',lambda *a: {})
    monkeypatch.setattr(reference,'verify_text',lambda *a: Report('inconclusive','No speech'))
    assert main(['verify-sync','movie.mkv',str(subtitle)]) == 3
    assert 'inconclusive' in capsys.readouterr().out
