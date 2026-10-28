# Subzero

Clean, convert, and check subtitles from the command line, or process a Jellyfin library through a persistent job queue.

[![CI](https://github.com/yuri-rod/subzero/actions/workflows/ci.yml/badge.svg)](https://github.com/yuri-rod/subzero/actions/workflows/ci.yml)

Remove sound descriptions and speaker labels from subtitles for the deaf and hard of hearing (SDH), while preserving dialogue and timestamps. For a media library, queue translation, timing audits, and caption recovery, then inspect progress through the CLI or authenticated API.

## Try it locally

Python 3.10 or newer is required. The core CLI has no runtime dependencies. From a checkout:

```console
git clone https://github.com/yuri-rod/subzero.git
cd subzero
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in PowerShell, then run:

```console
python -m pip install -e .
python -c "import shutil; shutil.copyfile('examples/dialogue.srt', 'demo.srt')"
subzero fix demo.srt --dry-run
subzero fix demo.srt --backup demo-backup
python -c "from pathlib import Path; print(Path('demo.srt').read_text(), end='')"
```

The synthetic example produces:

```srt
1
00:00:03,500 --> 00:00:05,000
- Ready?
- Ready.

```

The original is saved as `demo-backup/demo.srt`. `fix` modifies its input unless `--dry-run` is set. It writes a uniquely created temporary file in the same directory, preserves permission bits, and replaces the input only after writing the complete result. Existing backups are retained by basename; use a separate backup directory for sources that share a filename.

For the published CLI without a checkout, use `pip install subzero-cli` or `uv tool install subzero-cli`.

## Capabilities

- Clean SDH cues and repair collapsed dialogue turns without reflowing intact short lines.
- Convert SRT, WebVTT, ASS, and SSA; shift timestamps and extract embedded text subtitles with FFmpeg.
- Verify speech timing and attempt conservative offset or framerate corrections.
- Translate through an explicitly configured provider; recover burned-in English captions with Apple Vision on macOS.
- Inspect, cancel, and resume worker jobs through the CLI or bearer-authenticated HTTP API.

Install `subzero-cli[sync]` for timing analysis or `subzero-cli[worker]` for the daemon. Video workflows need FFmpeg and ffprobe. The toolkit is written in Python, with a small Swift bridge to Apple Vision for OCR. That feature needs macOS and the Swift compiler. Translation engines and transcription models are configured separately; the example above uses none of them.

See the [command and worker reference](docs/usage.md) for recipes, provider configuration, API requests, platform requirements, and service setup. `subzero COMMAND --help` shows the installed command's options.

## Implementation and recovery

```text
operator / Jellyfin
        |
        v
bearer-authenticated API -> SQLite queue -> single runner -> subtitle handlers
                                ^                |
                                +-- restart -----+
```

The worker atomically claims one job at a time. Startup requeues interrupted running jobs. Automatic jobs retry handler exceptions up to five times with backoff; manual jobs fail for inspection. Quota-paused jobs hold the queue until resumed or cancelled. Cancellation is cooperative: a cancelled state does not mean a native subprocess has already exited.

The repair workflow stages and validates candidates, checks whether its source changed, and backs up the installed subtitle before replacement. A later media-server refresh can fail after installation, so inspect `resultPath`, job details, and the installed file before retrying. Timing and language checks do not establish translation accuracy.

| Area | Implementation | Tests |
| --- | --- | --- |
| Cleanup and file replacement | [core.py](src/subzero/core.py) | [test_core.py](tests/test_core.py), [watcher backups](tests/test_watch.py) |
| Format conversion, timestamp shifts, bilingual merge | [convert.py](src/subzero/convert.py), [shift.py](src/subzero/shift.py), [merge.py](src/subzero/merge.py) | [conversion](tests/test_convert.py), [shifts](tests/test_shift.py), [merge](tests/test_merge.py) |
| Embedded subtitle extraction | [extract.py](src/subzero/extract.py) | [test_extract.py](tests/test_extract.py) |
| Speech timing and correction | [reference.py](src/subzero/reference.py), [timing.py](src/subzero/timing.py), [sync.py](src/subzero/sync.py) | [reference](tests/test_reference.py), [timing](tests/test_timing.py), [sync](tests/test_sync.py) |
| Translation and quality checks | [translate.py](src/subzero/translate.py), [translation_quality.py](src/subzero/translation_quality.py) | [translation](tests/test_translate.py), [quality checks](tests/test_translation_quality.py) |
| Caption recovery | [ocr.py](src/subzero/ocr.py), [Vision bridge](src/subzero/vision_ocr.swift) | [OCR](tests/test_ocr.py), [caption timing](tests/test_caption_timeline.py), [Vision bridge](tests/test_vision_ocr_swift.py) |
| Persistent queue, retries, cancellation | [jobs.py](src/subzero/worker/jobs.py) | [test_jobs.py](tests/worker/test_jobs.py) |
| Candidate validation and installation | [syncflow.py](src/subzero/worker/syncflow.py) | [test_syncflow.py](tests/worker/test_syncflow.py) |
| Worker API and authentication | [api.py](src/subzero/worker/api.py) | [test_api_auth.py](tests/worker/test_api_auth.py), [test_api_jobs.py](tests/worker/test_api_jobs.py) |
| Jellyfin, OpenSubtitles, and qBittorrent | [Jellyfin client](src/subzero/worker/jellyfin.py), [subtitle search](src/subzero/worker/opensubs.py), [completion watcher](src/subzero/worker/qbt_watch.py) | [Jellyfin](tests/worker/test_jellyfin.py), [OpenSubtitles](tests/worker/test_opensubs.py), [qBittorrent](tests/worker/test_qbt_watch.py) |
| Subtitle contribution | [contribute.py](src/subzero/worker/contribute.py) | [test_contribute.py](tests/worker/test_contribute.py) |

## Development and tests

In the activated checkout environment:

```console
python -m pip install -e ".[dev]"
python -m pytest
```

Run the recovery and file-safety tests without a media server or translation provider:

```console
python -m pytest tests/worker/test_jobs.py tests/worker/test_syncflow.py tests/test_core.py tests/test_watch.py
```

These tests use temporary files, SQLite databases, and mocked providers. CI runs on pushes and pull requests for every branch, with Python 3.10 through 3.13 on Linux, macOS, and Windows. Real media-server delivery, provider quality, and complete OCR scans need separate integration validation, tracked in [TODO.md](TODO.md).

For a contribution, reproduce the behavior with a focused test, preserve existing CLI and library contracts, and run the relevant suite. Reuse existing helpers and the standard library before adding dependencies. Keep optional integrations behind their existing extras. Write docs, comments, and user-facing messages in English. Release history is in [CHANGELOG.md](CHANGELOG.md).

[MIT license](LICENSE). Copyright 2026 Yuri Barreira. [Support the project](https://buymeacoffee.com/yurirod).
