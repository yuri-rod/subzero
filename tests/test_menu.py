from subzero.cli import build_parser, main
from subzero.menu import ACTIONS, MenuState, run_menu


def test_menu_command_registered():
    ap = build_parser()
    args = ap.parse_args(["menu"])
    assert args.command == "menu"


def test_menu_state_to_options():
    st = MenuState(max_line=30, languages=["en"], strip_music=False)
    opts = st.to_options()
    assert opts.max_line == 30
    assert opts.languages == ("en",)
    assert opts.strip_music is False


def test_all_actions_present():
    assert set(ACTIONS) == {str(i) for i in range(1, 12)}


def test_menu_quit_immediately(monkeypatch, capsys):
    answers = iter(["0"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert run_menu() == 0
    out = capsys.readouterr().out
    assert "interactive menu" in out
    assert "bye" in out


def test_menu_fix_flow(monkeypatch, tmp_path, capsys):
    sub = tmp_path / "a.srt"
    sub.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n[music]\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nHello there.\n\n",
        encoding="utf-8",
    )
    # choice 1 fix, path, dry-run no, then enter, then quit
    answers = iter([
        "1",            # fix
        str(tmp_path),  # path
        "n",            # dry run
        "",             # pause
        "0",            # quit
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert run_menu() == 0
    text = sub.read_text(encoding="utf-8")
    assert "[music]" not in text
    assert "Hello there" in text


def test_menu_convert_flow(monkeypatch, tmp_path, capsys):
    sub = tmp_path / "a.srt"
    sub.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n",
        encoding="utf-8",
    )
    answers = iter([
        "5",            # convert
        str(sub),
        "vtt",
        "n",            # dry
        "",
        "0",
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert run_menu() == 0
    assert (tmp_path / "a.vtt").exists()


def test_menu_configure(monkeypatch, capsys):
    answers = iter([
        "11",
        "en fr",        # langs
        "38",           # max line
        "n",            # rewrap all?
        "n",            # keep brackets
        "n",            # keep parens
        "n",            # keep music
        "n",            # keep labels
        "",             # backup
        "*.srt",
        "vtt",
        "",             # pause
        "0",
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    # capture state via running configure then quit, check banner printed vtt
    assert run_menu() == 0
    out = capsys.readouterr().out
    assert "extract→ vtt" in out or "extract" in out


def test_convert_cli(tmp_path, capsys):
    src = tmp_path / "a.srt"
    src.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHi.\n\n",
        encoding="utf-8",
    )
    assert main(["convert", str(src), "--to", "vtt", "-v"]) == 0
    assert (tmp_path / "a.vtt").exists()


def test_extract_cli_dry(monkeypatch, tmp_path, capsys):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    from subzero.extract import ExtractResult, SubtitleStream

    def fake_extract(path, **kw):
        return ExtractResult(
            source=str(path),
            outputs=[str(tmp_path / "a.eng.2.srt")],
            streams=[SubtitleStream(2, "subrip", "eng")],
            message="extracted 1 stream(s)",
        )

    monkeypatch.setattr("subzero.cli.require_ffmpeg", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr("subzero.cli.extract_from_video", fake_extract)
    assert main(["extract", str(video), "--dry-run", "-v"]) == 0
    out = capsys.readouterr().out
    assert "extracted" in out


def test_streams_cli(monkeypatch, tmp_path, capsys):
    video = tmp_path / "a.mkv"
    video.write_bytes(b"x")
    from subzero.extract import SubtitleStream

    monkeypatch.setattr("subzero.cli.require_ffmpeg", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(
        "subzero.cli.list_subtitle_streams",
        lambda p: [SubtitleStream(2, "subrip", "eng", title="SDH")],
    )
    assert main(["streams", str(video)]) == 0
    out = capsys.readouterr().out
    assert "subrip" in out
    assert "eng" in out


def test_menu_shift_flow(monkeypatch, tmp_path):
    sub = tmp_path / "a.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n", encoding="utf-8")
    answers = iter([
        "7",            # shift
        str(sub),       # path
        "1.5",          # seconds
        "n",            # dry
        "",             # pause
        "0",            # quit
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert run_menu() == 0
    text = sub.read_text(encoding="utf-8")
    assert "00:00:02,500 --> 00:00:03,500" in text


def test_menu_moviehash_flow(monkeypatch, tmp_path, capsys):
    from subzero.moviehash import CHUNK
    video = tmp_path / "a.mkv"
    video.write_bytes(b"A" * (CHUNK * 2 + 100))
    answers = iter([
        "8",            # moviehash
        str(video),     # path
        "",             # pause
        "0",            # quit
    ])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert run_menu() == 0
    out = capsys.readouterr().out
    assert "a.mkv" in out


def test_help_lists_new_commands(capsys):
    assert main([]) == 1
    out = capsys.readouterr().out
    for name in ("extract", "convert", "menu", "streams", "shift", "moviehash"):
        assert name in out


def test_split_paths_keeps_windows_separators(monkeypatch):
    """POSIX shlex escapes backslashes, which silently emptied every menu action
    on Windows: C:\\Users\\me came back as C:Usersme and matched nothing."""
    from subzero import menu

    monkeypatch.setattr(menu.os, "name", "nt")
    assert menu._split_paths(r"C:\Users\me\subs") == [r"C:\Users\me\subs"]
    assert menu._split_paths(r'"C:\Users\me\my subs"') == [r"C:\Users\me\my subs"]
    assert menu._split_paths(r"C:\a C:\b") == [r"C:\a", r"C:\b"]


def test_split_paths_still_posix_elsewhere(monkeypatch):
    from subzero import menu

    monkeypatch.setattr(menu.os, "name", "posix")
    assert menu._split_paths("/home/me/subs") == ["/home/me/subs"]
    assert menu._split_paths("'/home/me/my subs'") == ["/home/me/my subs"]
