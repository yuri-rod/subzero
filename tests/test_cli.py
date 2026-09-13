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


def test_contribute_parser_and_execution(monkeypatch):
    from subzero.cli import build_parser, main
    parser = build_parser()
    args = parser.parse_args(["contribute", "--dry-run", "--limit", "10", "--lang", "pt-BR"])
    assert args.dry_run is True
    assert args.limit == 10
    assert args.lang == "pt-BR"

    called = []
    def fake_contribute(argv):
        called.append(argv)
        return 0

    monkeypatch.setattr("subzero.worker.contribute.main", fake_contribute)
    rc = main(["contribute", "--dry-run", "--limit", "5", "--lang", "pt-BR", "--ledger", "test.db"])
    assert rc == 0
    assert "--dry-run" in called[0]
    assert "--limit" in called[0]
    assert "5" in called[0]
    assert "--lang" in called[0]
    assert "pt-BR" in called[0]
    assert "--ledger" in called[0]
    assert "test.db" in called[0]


def test_worker_contribute_action_parses():
    from subzero.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["worker", "contribute", "--port", "9000"])
    assert args.command == "worker"
    assert args.action == "contribute"
    assert args.port == 9000


@pytest.mark.parametrize("action", ["jobs", "sweep", "coverage", "audits"])
def test_worker_query_actions_parse(action):
    from subzero.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["worker", action])
    assert args.command == "worker"
    assert args.action == action



def test_fill_gaps_cli_parsing_and_alias():
    from subzero.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["fill-gaps", "vid.mkv", "sub.srt", "--to", "pt-BR", "--dry-run"])
    assert args.command == "fill-gaps"
    assert args.video == "vid.mkv"
    assert args.subtitle == "sub.srt"
    assert args.to == "pt-BR"
    assert args.dry_run is True

    alias_args = parser.parse_args(["ocr-sync", "vid.mkv", "sub.srt"])
    assert alias_args.command == "ocr-sync"
    assert alias_args.video == "vid.mkv"
    assert alias_args.subtitle == "sub.srt"

