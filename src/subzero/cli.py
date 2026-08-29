"""Command-line interface for Subzero."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .convert import FORMATS, convert_file
from .core import Options, analyze, fix_file, read
from .extract import (
    VIDEO_EXTENSIONS,
    collect_videos,
    extract_from_video,
    list_subtitle_streams,
    require_ffmpeg,
    ToolError,
)
from .merge import merge_files
from .moviehash import moviehash
from .roles import ROLES
from .shift import shift_file
from .sync import auto_sync_file
from .translate import translate_file
from .watch import INTERVAL, SETTLE, WORKERS, Watcher


def positive_int(v: str) -> int:
    n = int(v)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"{v} must be positive")
    return n


def non_negative_int(v: str) -> int:
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError(f"{v} cannot be negative")
    return n


def collect(paths: list[str], pattern: str, skip: str) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.rglob(pattern)))
        elif p.exists():
            out.append(p)
        else:
            print(f"subzero: no such file or directory: {p}", file=sys.stderr)
    if skip:
        import fnmatch
        out = [p for p in out if not fnmatch.fnmatch(p.name, skip)]
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def opts_from(args) -> Options:
    return Options(
        max_line=args.max_line,
        languages=tuple(args.lang),
        preserve_breaks=not args.rewrap_all,
        strip_brackets=not args.keep_brackets,
        strip_parens=not args.keep_parens,
        strip_music=not args.keep_music,
        strip_labels=not args.keep_labels,
    )


def cmd_fix(args) -> int:
    opts = opts_from(args)
    files = collect(args.paths, args.pattern, args.skip)
    if not files:
        print("subzero: nothing to do", file=sys.stderr)
        return 1
    changed = failed = total_cues = dropped_cues = rewrapped_cues = 0
    for p in files:
        try:
            res = fix_file(p, opts, backup_dir=args.backup, dry=args.dry_run)
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f"failed {p}: {str(e)[:160]}", file=sys.stderr)
            continue
        if res is None:
            failed += 1
            print(f"failed {p}: no cues parsed", file=sys.stderr)
            continue
        total_cues += res.cues
        dropped_cues += res.dropped
        rewrapped_cues += res.rewrapped
        if res.changed:
            changed += 1
            verb = "would fix" if args.dry_run else "fixed"
            if args.verbose or args.dry_run:
                print(f"{verb} {p} (cues={res.cues} sdh-{res.dropped} rewrap={res.rewrapped})")
    verb = "would change" if args.dry_run else "changed"
    print(f"{len(files)} files, {verb} {changed} (sdh-{dropped_cues} rewrap={rewrapped_cues}), failed {failed}")
    return 1 if failed else 0


def cmd_check(args) -> int:
    opts = opts_from(args)
    files = collect(args.paths, args.pattern, args.skip)
    if not files:
        print("subzero: nothing to do", file=sys.stderr)
        return 1
    dirty = unreadable = 0
    print(f"{'file':<48}{'cues':>6}{'sdh':>7}{'coll':>7}{'long':>6}  encoding")
    for p in files:
        try:
            srt, encoding = read(p)
        except Exception as e:                              # noqa: BLE001
            unreadable += 1
            print(f"{str(p)[:48]:<48}  unreadable: {e}", file=sys.stderr)
            continue
        st = analyze(srt, opts)
        bad = st.sdh or st.collapsed or st.long_lines or encoding != "utf-8"
        dirty += bool(bad)
        mark = "*" if bad else " "
        print(f"{mark}{str(p)[:47]:<47}{st.cues:>6}{st.pct('sdh'):>6}%{st.pct('collapsed'):>6}%{st.pct('long_lines'):>5}%  {encoding}")
    print(f"{len(files)} files, {dirty} need work, {unreadable} unreadable")
    return 1 if (dirty or unreadable) else 0


def cmd_shift(args) -> int:
    files = collect(args.paths, args.pattern, args.skip)
    if not files:
        print("subzero: nothing to do", file=sys.stderr)
        return 1
    failed = shifted = total_cues = 0
    delta = args.seconds or 0.0
    scale = args.scale or 1.0
    for p in files:
        out = None
        if args.output:
            out_path = Path(args.output)
            if len(files) == 1 and (not out_path.exists() or out_path.is_file()):
                out = out_path
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                out = out_path / p.name
        try:
            target, count = shift_file(
                p,
                delta_seconds=delta,
                scale_factor=scale,
                from_fps=args.from_fps,
                to_fps=args.to_fps,
                output=out,
                backup_dir=args.backup,
                dry=args.dry_run,
            )
            shifted += 1
            total_cues += count
            fps_info = f", fps={args.from_fps}->{args.to_fps}" if (args.from_fps and args.to_fps) else ""
            verb = "would shift" if args.dry_run else "shifted"
            if args.verbose or args.dry_run:
                print(f"{verb} {p.name} -> {target} ({count} cues, {delta:+.3f}s{fps_info})")
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f"failed {p}: {str(e)[:160]}", file=sys.stderr)
    verb = "would shift" if args.dry_run else "shifted"
    print(f"{len(files)} files, {verb} {shifted} ({total_cues} cues), failed {failed}")
    return 1 if failed else 0


def cmd_convert(args) -> int:
    files = collect(args.paths, args.pattern or "*", args.skip)
    if not files:
        print("subzero: nothing to do", file=sys.stderr)
        return 1
    target_fmt = args.to.lower().lstrip(".")
    failed = converted = total_cues = 0
    opts = opts_from(args) if args.fix else None

    for p in files:
        out = None
        if args.output:
            out_path = Path(args.output)
            if len(files) == 1 and (not out_path.exists() or out_path.is_file()):
                out = out_path
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                out = out_path / (p.stem + f".{target_fmt}")
        try:
            res = convert_file(p, target_fmt, source_fmt=args.from_fmt, output=out, dry=args.dry_run)
            converted += 1
            total_cues += res.cues
            verb = "would write" if args.dry_run else "wrote"
            fix_note = ""
            if args.fix and not args.dry_run and res.path:
                fix_res = fix_file(res.path, opts, backup_dir=args.backup)
                if fix_res and fix_res.changed:
                    fix_note = f" (fixed sdh-{fix_res.dropped} rewrap={fix_res.rewrapped})"
            if args.verbose or args.dry_run:
                print(f"{verb} {res.path} ({res.cues} cues, {res.source_format}->{res.target_format}){fix_note}")
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f"failed {p}: {str(e)[:160]}", file=sys.stderr)
    verb = "would convert" if args.dry_run else "converted"
    print(f"{len(files)} files, {verb} {converted} ({total_cues} cues), failed {failed}")
    return 1 if failed else 0


def cmd_extract(args) -> int:
    try:
        require_ffmpeg()
    except ToolError as e:
        print(f"subzero extract: {e}", file=sys.stderr)
        return 2

    videos = collect_videos(args.paths)
    if not videos:
        print("subzero extract: no video files found", file=sys.stderr)
        return 1

    opts = opts_from(args) if args.fix else None
    languages = tuple(args.language) if args.language else None
    indices = tuple(args.index) if args.index else None

    def fix_cb(path: Path) -> None:
        if opts:
            fix_file(path, opts, backup_dir=args.backup)

    failed = total_out = 0
    for v in videos:
        try:
            res = extract_from_video(
                v,
                output_dir=args.output,
                fmt=args.format,
                languages=languages,
                indices=indices,
                all_streams=args.all,
                dry=args.dry_run,
                fix=fix_cb if args.fix else None,
                fix_opts=opts,
            )
            total_out += len(res.outputs)
            if args.verbose or args.dry_run:
                print(f"{v.name}: {res.message}")
                for o in res.outputs:
                    print(f"  wrote {o}")
                for s in res.skipped:
                    print(f"  skipped bitmap/unsupported: {s}")
                for f in res.fixed:
                    print(f"  fixed {f}")
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f"failed {v}: {str(e)[:160]}", file=sys.stderr)

    print(f"{len(videos)} videos, extracted {total_out} streams, failed {failed}")
    return 1 if failed else 0


def cmd_streams(args) -> int:
    try:
        require_ffmpeg()
    except ToolError as e:
        print(f"subzero streams: {e}", file=sys.stderr)
        return 2
    videos = collect_videos(args.paths)
    if not videos:
        print("subzero streams: no video files found", file=sys.stderr)
        return 1
    for v in videos:
        print(f"{v}:")
        try:
            streams = list_subtitle_streams(v)
        except Exception as e:                              # noqa: BLE001
            print(f"  error: {e}", file=sys.stderr)
            continue
        if not streams:
            print("  (no subtitle streams found)")
            continue
        for s in streams:
            print(f"  {s.label}")
    return 0


def cmd_sync(args) -> int:
    try:
        require_ffmpeg()
    except ToolError as e:
        print(f"subzero sync: {e}", file=sys.stderr)
        return 2
    try:
        target, count, offset = auto_sync_file(
            args.video,
            args.subtitle,
            output=args.output,
            backup_dir=args.backup,
            dry=args.dry_run,
        )
        verb = "would sync" if args.dry_run else "synced"
        print(f"{verb} {args.subtitle} -> {target} ({count} cues, offset={offset:+.3f}s)")
        return 0
    except Exception as e:                                  # noqa: BLE001
        print(f"subzero sync error: {e}", file=sys.stderr)
        return 1


def cmd_translate(args) -> int:
    files = collect(args.paths, args.pattern, args.skip)
    if not files:
        print("subzero translate: nothing to do", file=sys.stderr)
        return 1
    failed = 0
    for p in files:
        try:
            out = translate_file(
                p,
                target_lang=args.to,
                output=args.output,
                provider=args.provider,
                model=args.model,
                url=args.url,
                api_key=args.api_key,
            )
            print(f"translated {p.name} -> {out.name} ({args.to})")
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f"failed {p}: {str(e)[:160]}", file=sys.stderr)
    return 1 if failed else 0


def cmd_merge(args) -> int:
    try:
        out, count = merge_files(
            args.primary,
            args.secondary,
            output=args.output,
            separator=args.separator.replace("\\n", "\n"),
            secondary_color=args.color,
        )
        print(f"merged {args.primary} + {args.secondary} -> {out} ({count} cues)")
        return 0
    except Exception as e:                                  # noqa: BLE001
        print(f"subzero merge error: {e}", file=sys.stderr)
        return 1


def cmd_moviehash(args) -> int:
    files = collect(args.paths, args.pattern, args.skip)
    if not files:
        print("subzero: no files found", file=sys.stderr)
        return 1
    for p in files:
        try:
            h = moviehash(p)
            print(f"{h}  {p}")
        except Exception as e:                              # noqa: BLE001
            print(f"failed {p}: {str(e)[:160]}", file=sys.stderr)
    return 0


def cmd_watch(args) -> int:
    roots = [Path(p) for p in args.paths]
    opts = opts_from(args)
    watcher = Watcher(
        roots=roots,
        opts=opts,
        backup_dir=args.backup,
        state_file=args.state,
        pattern=args.pattern,
        settle=args.settle,
        workers=args.workers,
        on_event=print if args.verbose else lambda _: None,
    )
    try:
        watcher.run(interval=args.interval, once=args.once)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_menu(args) -> int:                                  # noqa: ARG001
    from .menu import run_menu
    return run_menu()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="subzero",
        description="Universal subtitle & audio AI toolkit: clean SDH, auto-sync, shift, convert, extract, translate, and merge.",
    )
    ap.add_argument("--version", action="store_true", help="print version and exit")
    sub = ap.add_subparsers(dest="command")

    def shared(p, with_paths: bool = True):
        if with_paths:
            p.add_argument("paths", nargs="+", help="files or directories")
        p.add_argument("--lang", nargs="+", default=["en", "pt"],
                       choices=sorted(ROLES), metavar="CODE",
                       help=f"languages for lower-case role labels ({', '.join(sorted(ROLES))})")
        p.add_argument("--max-line", type=positive_int, default=42,
                       help="characters per line before a cue is re-broken (default 42)")
        p.add_argument("--pattern", default="*.srt", help="glob for directory walks")
        p.add_argument("--skip", default="", help="glob of filenames to leave alone")
        p.add_argument("--rewrap-all", action="store_true",
                       help="re-break every cue, even ones already broken by hand")
        p.add_argument("--keep-brackets", action="store_true", help="keep [sound cues]")
        p.add_argument("--keep-parens", action="store_true", help="keep (sound cues)")
        p.add_argument("--keep-music", action="store_true", help="keep music symbols")
        p.add_argument("--keep-labels", action="store_true", help="keep SPEAKER: labels")
        p.add_argument("-v", "--verbose", action="store_true")

    fix = sub.add_parser("fix", help="strip SDH cues and format dialogue in place")
    shared(fix)
    fix.add_argument("--dry-run", action="store_true", help="report without writing")
    fix.add_argument("--backup", metavar="DIR", help="copy each original here first")
    fix.set_defaults(func=cmd_fix)

    check = sub.add_parser("check", help="report subtitle defects without changing files")
    shared(check)
    check.set_defaults(func=cmd_check)

    shift = sub.add_parser("shift", help="shift and/or speed-convert subtitle timestamps")
    shift.add_argument("paths", nargs="+", help="subtitle files or directories")
    shift.add_argument("--seconds", "-s", type=float, default=0.0, help="seconds to shift (e.g. +1.5 or -2.0)")
    shift.add_argument("--scale", "--factor", type=float, default=1.0, help="speed multiplier (e.g. 1.0427)")
    shift.add_argument("--from-fps", help="source framerate (e.g. 23.976, 25, 24)")
    shift.add_argument("--to-fps", help="target framerate (e.g. 25, 23.976, 24)")
    shift.add_argument("-o", "--output", metavar="PATH", help="output file or directory")
    shift.add_argument("--pattern", default="*.srt", help="glob for directory walks")
    shift.add_argument("--skip", default="", help="glob of filenames to leave alone")
    shift.add_argument("--backup", metavar="DIR", help="backup original before shift")
    shift.add_argument("--dry-run", action="store_true", help="report without writing")
    shift.add_argument("-v", "--verbose", action="store_true")
    shift.set_defaults(func=cmd_shift)

    conv = sub.add_parser("convert", help="convert between subtitle formats (srt, vtt, ass, ssa, sub)")
    shared(conv)
    conv.add_argument("--to", required=True, metavar="FMT", help=f"target format: {', '.join(FORMATS)}")
    conv.add_argument("--from", dest="from_fmt", default=None, metavar="FMT", help="source format (default: auto)")
    conv.add_argument("-o", "--output", metavar="PATH", help="output file or directory")
    conv.add_argument("--dry-run", action="store_true", help="report without writing")
    conv.add_argument("--fix", action="store_true", help="run SDH cleanup after conversion")
    conv.add_argument("--backup", metavar="DIR", help="backup originals before fix")
    conv.set_defaults(func=cmd_convert)

    ext = sub.add_parser("extract", help="extract soft subtitles from video (mp4, mkv, mov, mlv, …)")
    shared(ext)
    ext.add_argument("--format", default="srt", metavar="FMT", help=f"output format (default srt; {', '.join(FORMATS)})")
    ext.add_argument("-o", "--output", metavar="DIR", help="directory for extracted files")
    ext.add_argument("--language", nargs="+", metavar="CODE", help="only extract streams matching languages")
    ext.add_argument("--index", nargs="+", type=non_negative_int, metavar="N", help="stream index(es) to extract")
    ext.add_argument("--all", action="store_true", help="extract all matching streams")
    ext.add_argument("--fix", action="store_true", help="run SDH cleanup on extracted subtitles")
    ext.add_argument("--backup", metavar="DIR", help="backup before fix")
    ext.add_argument("--dry-run", action="store_true", help="list actions only")
    ext.set_defaults(func=cmd_extract)

    streams = sub.add_parser("streams", help="list subtitle streams in a video container")
    streams.add_argument("paths", nargs="+", help="video files or directories")
    streams.set_defaults(func=cmd_streams)

    sync = sub.add_parser("sync", help="auto-sync subtitle to video audio/speech delay")
    sync.add_argument("video", help="video file path")
    sync.add_argument("subtitle", help="subtitle file path")
    sync.add_argument("-o", "--output", metavar="PATH", help="output subtitle path")
    sync.add_argument("--backup", metavar="DIR", help="backup directory")
    sync.add_argument("--dry-run", action="store_true", help="report without writing")
    sync.set_defaults(func=cmd_sync)

    trans = sub.add_parser("translate", help="translate subtitle file using Ollama or OpenAI-compatible LLMs")
    trans.add_argument("paths", nargs="+", help="subtitle files or directories")
    trans.add_argument("--to", default="pt-BR", help="target language (default pt-BR)")
    trans.add_argument("-o", "--output", metavar="PATH", help="output file or directory")
    trans.add_argument("--provider", default="ollama", choices=["ollama", "openai", "openrouter", "groq", "deepseek"], help="LLM provider (default: ollama)")
    trans.add_argument("--model", default=None, help="LLM model name (defaults to provider recommendation)")
    trans.add_argument("--url", default=None, help="custom API base URL / Ollama host")
    trans.add_argument("--api-key", default=None, help="API key for cloud LLM providers")
    trans.add_argument("--pattern", default="*.srt", help="glob for directory walks")
    trans.add_argument("--skip", default="", help="glob of filenames to leave alone")
    trans.set_defaults(func=cmd_translate)

    merge = sub.add_parser("merge", help="merge two subtitles into dual-language/bilingual subtitles")
    merge.add_argument("primary", help="primary language subtitle file")
    merge.add_argument("secondary", help="secondary language subtitle file")
    merge.add_argument("-o", "--output", metavar="PATH", help="output subtitle file path")
    merge.add_argument("--separator", default="\n", help="separator between languages (default: newline)")
    merge.add_argument("--color", default=None, help="optional HTML color for secondary language (e.g. #ffff00)")
    merge.set_defaults(func=cmd_merge)

    mhash = sub.add_parser("moviehash", help="compute OpenSubtitles 64-bit file hash")
    mhash.add_argument("paths", nargs="+", help="files or directories")
    mhash.add_argument("--pattern", default="*", help="glob for directory walks")
    mhash.add_argument("--skip", default="", help="glob of filenames to leave alone")
    mhash.set_defaults(func=cmd_moviehash)

    watch = sub.add_parser("watch", help="clean subtitles automatically in background")
    shared(watch)
    watch.add_argument("--interval", type=positive_int, default=INTERVAL, help="seconds between sweeps")
    watch.add_argument("--settle", type=non_negative_int, default=SETTLE, help="ignore files modified recently")
    watch.add_argument("--workers", type=positive_int, default=WORKERS, help="threads per sweep")
    watch.add_argument("--state", metavar="FILE", help="state record file")
    watch.add_argument("--backup", metavar="DIR", help="backup directory")
    watch.add_argument("--once", action="store_true", help="one sweep and exit")
    watch.set_defaults(func=cmd_watch)

    menu = sub.add_parser("menu", help="interactive terminal menu for all features")
    menu.set_defaults(func=cmd_menu)

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.version:
        print(f"subzero {__version__}")
        return 0
    if not args.command:
        ap.print_help(sys.stdout)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
