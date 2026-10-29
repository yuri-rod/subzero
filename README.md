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

## Architecture and reliability

```text
operator / Jellyfin
        |
        v
bearer-authenticated API -> SQLite queue -> single runner -> subtitle handlers
                                ^                |
                                +-- restart -----+
```

### Transactional queue and recovery

Job routes use shared bearer-token authentication with `hmac.compare_digest` in [`require_token`](src/subzero/worker/api.py). [`JobStore.claim`](src/subzero/worker/jobs.py) uses `BEGIN IMMEDIATE` to atomically claim one job. Startup calls `requeue_running` for interrupted jobs. Separately, automatic handler failures retry up to five times with exponential backoff; manual failures stop for inspection. Quota-paused jobs hold the queue. Cancellation is cooperative and does not prove a native child has exited.

### Staged installation

Repair workflows retain candidates through [`SyncFlow.stage`](src/subzero/worker/syncflow.py), then validate before `install`. Installation checks job and source state, backs up existing subtitles, and writes through a temporary file, `fsync`, and atomic replacement or creation. Basic delivery uses [`tracks.deliver`](src/subzero/worker/tracks.py) and writes directly. A later audit or media-server refresh can fail after installation; inspect the installed file and job details before retrying.

### Local and API inference paths

Queued translation selects `TRANSLATION_PROVIDER=ollama` (the default) or `openai` in [`Config`](src/subzero/worker/config.py). The worker uses [`tracks.Ollama`](src/subzero/worker/tracks.py) or [`OpenAIClient`](src/subzero/translate.py); CLI translation uses `OllamaClient` or `OpenAIClient`. OpenAI-compatible endpoints include local `mlx_lm.server`, `llama-server`, vLLM, and remote services. Endpoint, model, and prompt settings enter the cache identity; API keys do not. External API servers retain their own lifecycle. Direct Apple Silicon speech transcription uses [`MlxModel.transcribe`](src/subzero/worker/mlx_asr.py) with `mlx-whisper`; Ollama can separately serve MLX-backed translation models. See [provider configuration](docs/usage.md#openai-compatible-worker-translation).

### Deterministic validation and retries

[`_parse_lines` and `_translate_lines`](src/subzero/translate.py) validate JSON entries or numbered lines, sequential cue IDs, nonempty text, and cue counts. Structural failures can trigger retries and recursive batch bisection while preserving cue order. [`check_language_completeness` and `check_excellence_guards`](src/subzero/worker/guards.py) check Portuguese language completeness, SDH artifacts, and formatting; `sanitize_to_excellence` applies cleanup. [`AppleFM`](src/subzero/worker/applefm.py) supports explicitly configured unit-level fallback for refusals or repeated echoes.

### Batched LLM-as-judge QA

For full worker translations into Portuguese, [`semantic_gate`](src/subzero/worker/syncflow.py) is enabled by default. [`check_translation` and `judge_pairs`](src/subzero/translation_quality.py) sample up to 40 source/output pairs and score adequacy through Ollama in sequential batches of 20. Mean adequacy below 3.5/5, at least two critical judgments, or at least five deterministic suspect cues rejects a candidate. The verdict identifies defective cue indexes; one bounded repair pass retranslates those cues individually, preserves untouched cues and timing, invalidates affected cache blocks, and re-evaluates with repaired cues included in the sample. A failed repair stops delivery. Initial judge outages fall back to deterministic checks; an outage after a scored rejection cannot approve its repair. Standalone CLI translation skips this gate.

### Compute leases and resumable caches

An optional macOS policy uses [`compute_phase`](src/subzero/compute.py) to combine a thread lock with POSIX `flock` and manage Ollama's service lifecycle. Participating native children inherit the lease through `compute_lease_fd` and [`pass_fds`](src/subzero/worker/asr_process.py), retaining ownership if the parent crashes. On `ComputeShutdownError`, the phase retains the lock and `_poison` records the blocked state. Coordination covers participating processes using the same policy.

[`_translate_cues_cached` and `_repair_translation`](src/subzero/worker/syncflow.py) key translation caches by source hashes and provider/model/prompt settings. [`read_cached_block` and `write_cached_block`](src/subzero/worker/translate_cache.py) verify source, output digest, timing, and cue count, and persist blocks with `fsync` and atomic replacement for resume. [`build_reference`](src/subzero/reference.py) caches timing references using `fingerprint`, which hashes video metadata, validator version, and sampled file bytes.

### Memory and swap admission

[`PressureWatchdog`](src/subzero/compute.py) samples macOS `vm_stat`/`sysctl`, Linux `/proc/meminfo`, and NVIDIA free VRAM. Configurable byte thresholds block new compute phases and defer queued jobs with `phase=resource_pressure` and a retry time, without consuming handler retries. `/health` exposes the snapshot and blocking reason. Under a managed compute policy, the watchdog acquires the idle lease before stopping Ollama; active leases remain untouched. Admission resumes when pressure clears. Thresholds default to disabled. See [pressure configuration](docs/usage.md#resource-pressure-and-deferred-jobs).

### Quantization benchmarks

[`subzero benchmark`](src/subzero/benchmark.py) runs a fixed subtitle workload through Ollama or OpenAI-compatible servers and writes JSON plus Markdown: p50/p95 workload latency, cues/sec, reported completion tokens/sec, sampled host memory/swap, deterministic guardrail rate, and optional judge adequacy. Reports retain outputs, failures, model/workload hashes, and missing metrics. The [BF16 versus Q8_0 baseline](docs/benchmarks/README.md) includes measured results, runtime settings, and reproduction steps. The harness uses the standard library and adds no base CLI dependencies.

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
| Queued providers and semantic repair | [translate.py](src/subzero/translate.py), [syncflow.py](src/subzero/worker/syncflow.py) | [queued providers](tests/worker/test_worker_providers.py), [targeted repair](tests/worker/test_semantic_retry.py) |
| Resource admission and benchmarking | [compute.py](src/subzero/compute.py), [benchmark.py](src/subzero/benchmark.py) | [pressure controls](tests/test_compute_pressure.py), [benchmark harness](tests/test_benchmark.py) |
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
