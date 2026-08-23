"""Interactive terminal menu for subzero.

No third-party UI library, plain stdin prompts so the tool stays dependency-free.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .convert import FORMATS, convert_file, detect_format
from .core import Options, analyze, fix_file, read
from .extract import (
    VIDEO_EXTENSIONS,
    collect_videos,
    extract_from_video,
    list_subtitle_streams,
    require_ffmpeg,
    ToolError,
)
from .moviehash import moviehash
from .roles import ROLES
from .shift import shift_file
from .watch import INTERVAL, SETTLE, WORKERS, Watcher


@dataclass
class MenuState:
    """Options the user tunes in the menu; applied to subsequent actions."""

    max_line: int = 42
    languages: list[str] = field(default_factory=lambda: ["en", "pt"])
    preserve_breaks: bool = True
    strip_brackets: bool = True
    strip_parens: bool = True
    strip_music: bool = True
    strip_labels: bool = True
    backup_dir: str | None = None
    pattern: str = "*.srt"
    extract_format: str = "srt"
    last_path: str = ""

    def to_options(self) -> Options:
        return Options(
            max_line=self.max_line,
            languages=tuple(self.languages),
            preserve_breaks=self.preserve_breaks,
            strip_brackets=self.strip_brackets,
            strip_parens=self.strip_parens,
            strip_music=self.strip_music,
            strip_labels=self.strip_labels,
        )


def _print(msg: str = "", file=None) -> None:
    print(msg, file=file or sys.stdout, flush=True)


def _input(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        _print()
        return ""
    if not value and default is not None:
        return default
    return value


def _yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    ans = _input(f"{prompt} ({hint})", "y" if default else "n").lower()
    if not ans:
        return default
    return ans in {"y", "yes", "1", "true"}


def _pause() -> None:
    _input("Press Enter to continue", "")


def _banner(state: MenuState) -> None:
    _print()
    _print("=" * 60)
    _print(f"  subzero {__version__}: interactive menu")
    _print("=" * 60)
    langs = ",".join(state.languages)
    flags = []
    if not state.strip_brackets:
        flags.append("keep-brackets")
    if not state.strip_parens:
        flags.append("keep-parens")
    if not state.strip_music:
        flags.append("keep-music")
    if not state.strip_labels:
        flags.append("keep-labels")
    if not state.preserve_breaks:
        flags.append("rewrap-all")
    flag_s = f"  flags: {', '.join(flags)}" if flags else "  flags: defaults"
    _print(f"  langs: {langs}   max-line: {state.max_line}   extract→ {state.extract_format}")
    _print(flag_s)
    if state.backup_dir:
        _print(f"  backup: {state.backup_dir}")
    if state.last_path:
        _print(f"  last path: {state.last_path}")
    _print("-" * 60)


def _menu_table() -> None:
    _print("  1) Fix SDH in subtitle files")
    _print("  2) Check subtitle files (report only)")
    _print("  3) Extract subtitles from video (mp4/mkv/mlv/...)")
    _print("  4) Extract from video + fix SDH")
    _print("  5) Convert subtitle format (srt/vtt/ass)")
    _print("  6) Convert + fix SDH")
    _print("  7) Shift subtitle timestamps (+/- seconds)")
    _print("  8) Calculate OpenSubtitles MovieHash")
    _print("  9) List subtitle streams in a video")
    _print(" 10) Watch a directory for new subtitles")
    _print(" 11) Configure options")
    _print("  0) Exit")
    _print("-" * 60)


def _ask_paths(state: MenuState, kind: str = "subtitle") -> list[str]:
    hint = state.last_path or "."
    raw = _input(f"Path(s) to {kind} file(s) or directory", hint)
    if not raw:
        return []
    paths = shlex.split(raw)
    if paths:
        state.last_path = paths[0]
    return paths


def _collect_subs(paths: list[str], pattern: str) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.rglob(pattern)))
            # also pick converted formats when pattern is default srt
            if pattern == "*.srt":
                for alt in ("*.vtt", "*.ass", "*.ssa"):
                    found.extend(sorted(p.rglob(alt)))
        elif p.exists():
            found.append(p)
        else:
            _print(f"  no such path: {p}", file=sys.stderr)
    # unique preserve order
    seen = set()
    out = []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def action_fix(state: MenuState) -> None:
    paths = _ask_paths(state, "subtitle")
    if not paths:
        return
    dry = _yes("Dry run (report only, no write)?", False)
    files = _collect_subs(paths, state.pattern)
    if not files:
        _print("  nothing to do")
        return
    opts = state.to_options()
    changed = failed = 0
    for p in files:
        # convert non-srt to srt first for fixing
        target = p
        if p.suffix.lower() in {".vtt", ".ass", ".ssa"}:
            try:
                conv = convert_file(p, "srt")
                target = Path(conv.path)
                _print(f"  converted {p.name} → {target.name}")
            except Exception as e:  # noqa: BLE001
                _print(f"  failed convert {p.name}: {e}")
                failed += 1
                continue
        try:
            res = fix_file(target, opts, backup_dir=state.backup_dir, dry=dry)
        except Exception as e:  # noqa: BLE001
            _print(f"  failed {p.name}: {e}")
            failed += 1
            continue
        if res is None:
            _print(f"  failed {p.name}: no cues parsed")
            failed += 1
            continue
        if res.changed:
            changed += 1
            verb = "would fix" if dry else "fixed"
            _print(f"  {verb} {target.name} cues={res.cues} sdh-{res.dropped} rewrap={res.rewrapped}")
        else:
            _print(f"  clean {target.name}")
    _print(f"  done: {len(files)} files, changed {changed}, failed {failed}")


def action_check(state: MenuState) -> None:
    paths = _ask_paths(state, "subtitle")
    if not paths:
        return
    files = _collect_subs(paths, state.pattern)
    if not files:
        _print("  nothing to do")
        return
    opts = state.to_options()
    dirty = 0
    _print(f"  {'file':<40}{'cues':>6}{'sdh':>7}{'coll':>7}{'long':>6}")
    for p in files:
        try:
            if p.suffix.lower() != ".srt":
                from .convert import convert_text
                raw = p.read_text(encoding="utf-8", errors="replace")
                fmt = detect_format(p, raw)
                srt = convert_text(raw, "srt", source=fmt).text
                encoding = "utf-8"
            else:
                srt, encoding = read(p)
        except Exception as e:  # noqa: BLE001
            _print(f"  {p.name[:40]:<40}  unreadable: {e}")
            dirty += 1
            continue
        st = analyze(srt, opts)
        bad = st.sdh or st.collapsed or st.long_lines or encoding != "utf-8"
        dirty += bool(bad)
        mark = "*" if bad else " "
        _print(
            f"  {mark}{p.name[:39]:<39}{st.cues:>6}{st.pct('sdh'):>6}%"
            f"{st.pct('collapsed'):>6}%{st.pct('long_lines'):>5}%"
        )
    _print(f"  {len(files)} files, {dirty} need work")


def _extract_common(state: MenuState, do_fix: bool) -> None:
    try:
        require_ffmpeg()
    except ToolError as e:
        _print(f"  {e}")
        return
    paths = _ask_paths(state, "video")
    if not paths:
        return
    videos = collect_videos(paths)
    if not videos:
        _print("  no video files found "
               f"(extensions: {', '.join(VIDEO_EXTENSIONS[:8])}…)")
        return
    fmt = _input("Output subtitle format (srt/vtt/ass)", state.extract_format).lower()
    if fmt not in FORMATS:
        _print(f"  unsupported format {fmt}, using srt")
        fmt = "srt"
    state.extract_format = fmt
    out_dir = _input("Output directory (empty = next to video)", "") or None
    lang_raw = _input("Language filter (space-separated, empty = default first)", "")
    languages = tuple(lang_raw.split()) if lang_raw else None
    all_streams = _yes("Extract all matching streams?", False)
    dry = _yes("Dry run?", False)
    opts = state.to_options()

    def _fix(path, o=None):
        return fix_file(path, o or opts, backup_dir=state.backup_dir)

    for video in videos:
        _print(f"  → {video.name}")
        try:
            res = extract_from_video(
                video,
                output_dir=out_dir,
                fmt=fmt,
                languages=languages,
                all_streams=all_streams,
                dry=dry,
                fix=_fix if do_fix and not dry else None,
                fix_opts=opts if do_fix else None,
            )
        except Exception as e:  # noqa: BLE001
            _print(f"    error: {e}")
            continue
        _print(f"    {res.message}")
        for o in res.outputs:
            _print(f"    wrote {o}")
        for s in res.skipped:
            _print(f"    skip {s}")
        for f in res.fixed:
            _print(f"    fixed {f}")


def action_extract(state: MenuState) -> None:
    _extract_common(state, do_fix=False)


def action_extract_fix(state: MenuState) -> None:
    _extract_common(state, do_fix=True)


def action_convert(state: MenuState, do_fix: bool = False) -> None:
    paths = _ask_paths(state, "subtitle")
    if not paths:
        return
    target = _input("Target format (srt/vtt/ass)", "srt").lower()
    if target not in FORMATS:
        _print(f"  unsupported format: {target}")
        return
    dry = _yes("Dry run?", False)
    opts = state.to_options()
    files = _collect_subs(paths, "*")
    # filter to known subtitle extensions
    files = [
        p for p in files
        if p.suffix.lower() in {".srt", ".vtt", ".ass", ".ssa", ".webvtt"}
    ]
    if not files:
        _print("  nothing to convert")
        return
    for p in files:
        try:
            res = convert_file(p, target, dry=dry)
        except Exception as e:  # noqa: BLE001
            _print(f"  failed {p.name}: {e}")
            continue
        verb = "would write" if dry else "wrote"
        _print(f"  {verb} {res.path} ({res.cues} cues, {res.source_format}→{res.target_format})")
        if do_fix and not dry and target == "srt" and res.path:
            fr = fix_file(res.path, opts, backup_dir=state.backup_dir)
            if fr and fr.changed:
                _print(f"    fixed sdh-{fr.dropped} rewrap={fr.rewrapped}")
        elif do_fix and not dry and res.path and target != "srt":
            # convert to srt, fix, done
            srt = convert_file(res.path, "srt")
            fr = fix_file(srt.path, opts, backup_dir=state.backup_dir)
            if fr and fr.changed:
                _print(f"    fixed {srt.path} sdh-{fr.dropped} rewrap={fr.rewrapped}")


def action_shift(state: MenuState) -> None:
    paths = _ask_paths(state, "subtitle")
    if not paths:
        return
    raw = _input("Seconds to shift (e.g. +1.5 or -2.0)", "1.0")
    try:
        delta = float(raw)
    except ValueError:
        _print("  invalid float seconds")
        return
    dry = _yes("Dry run?", False)
    files = _collect_subs(paths, state.pattern)
    if not files:
        _print("  nothing to shift")
        return
    for p in files:
        try:
            target, count = shift_file(
                p,
                delta_seconds=delta,
                backup_dir=state.backup_dir,
                dry=dry,
            )
            verb = "would shift" if dry else "shifted"
            _print(f"  {verb} {p.name} -> {target.name} ({count} cues, {delta:+.3f}s)")
        except Exception as e:  # noqa: BLE001
            _print(f"  failed {p.name}: {e}")


def action_moviehash(state: MenuState) -> None:
    paths = _ask_paths(state, "video or subtitle")
    if not paths:
        return
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend([f for f in sorted(p.rglob("*")) if f.is_file()])
        elif p.is_file():
            files.append(p)
    if not files:
        _print("  no files found")
        return
    for p in files:
        try:
            h = moviehash(p)
            _print(f"  {h}  {p.name}")
        except Exception as e:  # noqa: BLE001
            _print(f"  failed {p.name}: {e}")


def action_list_streams(state: MenuState) -> None:
    try:
        require_ffmpeg()
    except ToolError as e:
        _print(f"  {e}")
        return
    paths = _ask_paths(state, "video")
    if not paths:
        return
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            videos = collect_videos([p])
        else:
            videos = [p]
        for video in videos:
            _print(f"  {video}")
            try:
                streams = list_subtitle_streams(video)
            except Exception as e:  # noqa: BLE001
                _print(f"    error: {e}")
                continue
            if not streams:
                _print("    (no subtitle streams)")
                continue
            for s in streams:
                _print(f"    {s.label}")


def action_watch(state: MenuState) -> None:
    paths = _ask_paths(state, "directory")
    if not paths:
        return
    once = _yes("Run a single sweep then return?", True)
    interval = INTERVAL
    if not once:
        raw = _input("Interval seconds", str(INTERVAL))
        try:
            interval = max(1, int(raw))
        except ValueError:
            interval = INTERVAL
    roots = [Path(p) for p in paths]
    watcher = Watcher(
        roots=roots,
        opts=state.to_options(),
        backup_dir=state.backup_dir,
        pattern=state.pattern,
        settle=0 if once else SETTLE,
        workers=WORKERS,
        on_event=lambda m: _print(f"  {m}"),
    )
    _print("  watching… (Ctrl+C to stop)" if not once else "  sweeping…")
    try:
        watcher.run(interval=interval, once=once)
    except KeyboardInterrupt:
        _print("  stopped")


def action_configure(state: MenuState) -> None:
    _print("  Configure options (empty keeps current)")
    raw = _input("Languages (space-separated)", " ".join(state.languages))
    if raw:
        langs = [x for x in raw.split() if x in ROLES]
        unknown = [x for x in raw.split() if x not in ROLES]
        if unknown:
            _print(f"  unknown ignored: {', '.join(unknown)} "
                   f"(known: {', '.join(sorted(ROLES))})")
        if langs:
            state.languages = langs
    raw = _input("Max line length", str(state.max_line))
    if raw:
        try:
            v = int(raw)
            if v >= 1:
                state.max_line = v
        except ValueError:
            _print("  invalid number")
    state.preserve_breaks = not _yes(
        "Rewrap all cues (not only broken ones)?", not state.preserve_breaks
    )
    state.strip_brackets = not _yes("Keep [bracket] cues?", not state.strip_brackets)
    state.strip_parens = not _yes("Keep (paren) cues?", not state.strip_parens)
    state.strip_music = not _yes("Keep music symbols?", not state.strip_music)
    state.strip_labels = not _yes("Keep SPEAKER: labels?", not state.strip_labels)
    bak = _input("Backup directory (empty = none)", state.backup_dir or "")
    state.backup_dir = bak or None
    state.pattern = _input("Subtitle glob pattern", state.pattern) or state.pattern
    fmt = _input("Default extract format", state.extract_format).lower()
    if fmt in FORMATS:
        state.extract_format = fmt
    _print("  options saved for this session")


ACTIONS = {
    "1": ("Fix SDH", action_fix),
    "2": ("Check", action_check),
    "3": ("Extract", action_extract),
    "4": ("Extract + fix", action_extract_fix),
    "5": ("Convert", lambda s: action_convert(s, do_fix=False)),
    "6": ("Convert + fix", lambda s: action_convert(s, do_fix=True)),
    "7": ("Shift timestamps", action_shift),
    "8": ("Calculate MovieHash", action_moviehash),
    "9": ("List streams", action_list_streams),
    "10": ("Watch", action_watch),
    "11": ("Configure", action_configure),
}


def run_menu(argv=None) -> int:  # noqa: ARG001
    """Entry point for ``subzero menu``."""
    state = MenuState()
    _print(f"subzero {__version__} interactive menu: type a number, or q to quit")
    while True:
        _banner(state)
        _menu_table()
        choice = _input("Choice", "0").lower()
        if choice in {"0", "q", "quit", "exit"}:
            _print("bye")
            return 0
        action = ACTIONS.get(choice)
        if not action:
            _print("  unknown choice")
            continue
        name, fn = action
        _print(f"\n» {name}")
        try:
            fn(state)
        except KeyboardInterrupt:
            _print("\n  cancelled")
        except Exception as e:  # noqa: BLE001
            _print(f"  error: {e}")
        _pause()
