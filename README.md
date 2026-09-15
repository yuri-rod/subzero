# Subzero

```
  ███████╗██╗   ██╗██████╗ ███████╗███████╗██████╗  ██████╗ 
  ██╔════╝██║   ██║██╔══██╗╚══███╔╝██╔════╝██╔══██╗██╔═══██╗
  ███████╗██║   ██║██████╔╝  ███╔╝ █████╗  ██████╔╝██║   ██║
  ╚════██║██║   ██║██╔══██╗ ███╔╝  ██╔══╝  ██╔══██╗██║   ██║
  ███████║╚██████╔╝██████╔╝███████╗███████╗██║  ██║╚██████╔╝
  ╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ 
```

Subtitle cleanup, timing verification, translation, and burned-in caption recovery.

[![Version](https://img.shields.io/badge/version-1.16.2-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Core dependencies: Zero](https://img.shields.io/badge/core_dependencies-zero-brightgreen.svg)](pyproject.toml)
[![Platform: Linux | macOS | Windows](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()
[![CI](https://github.com/yuri-rod/subzero/actions/workflows/ci.yml/badge.svg)](https://github.com/yuri-rod/subzero/actions/workflows/ci.yml)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-yellow.svg?style=flat&logo=buy-me-a-coffee)](https://buymeacoffee.com/yurirod)

---

## What it does

Subzero provides a CLI, Python library, directory watcher, and subtitle worker for Jellyfin and YUCAST.

* Removes SDH sound cues, speaker labels, and formatting tags, then repairs collapsed dialogue turns and long lines.
* Extracts embedded text subtitles and converts between SRT, WebVTT, ASS, and SSA.
* Verifies subtitle timing against local speech detection and applies supported constant-offset or framerate corrections.
* Recovers burned-in English captions with Apple Vision on macOS, including quiet passages missed by speech detection.
* Translates locally through Ollama or the worker's isolated LibreTranslate CPU engine, or through the worker's explicitly selected DeepL Free API. The CLI also supports explicitly selected OpenAI-compatible endpoints.
* Matches OpenSubtitles downloads, validates candidates, and stages replacements before installing worker output with backups.

The core uses the Python standard library. Timing analysis and the worker have optional Python dependencies; video operations need FFmpeg.

See [TODO.md](TODO.md) for known issues and remaining validation.

---

### Before and After Subzero

```srt
1
00:00:01,000 --> 00:00:03,000
[tense dramatic music playing]

2
00:00:03,500 --> 00:00:05,000
- MITCH: Yes, sir. - MAN: So I have a question.
```

**After `subzero fix show.srt`:**

```srt
1
00:00:03,500 --> 00:00:05,000
- Yes, sir.
- So I have a question.
```

---

## Quick Install

### Using `pip`

```console
pip install subzero-cli
```

### Using `uv` (Recommended)

```console
uv tool install subzero-cli
```

### Run Directly with `uvx` (No installation needed)

```console
uvx subzero-cli menu
```

Python 3.10+ is required. Install `ffmpeg` and `ffprobe` for extraction, timing analysis, transcription, or OCR. Native OCR also requires macOS and `swiftc`; the packaged Swift source is compiled locally on first use.

Install the extra dependencies for the features you use:

```console
pip install 'subzero-cli[sync]'
pip install 'subzero-cli[worker]'
```

With `uv tool`, use `uv tool install 'subzero-cli[worker,sync]'` for both extras. Translation engines and their models are separate installations; see the worker configuration below.

---

## Interactive Terminal Menu

Run `subzero menu` to choose cleanup, extraction, conversion, timestamp shifting, and directory watching interactively. Use `subzero COMMAND --help` for the current options of each command.

---

## Command Reference and Recipes

### 1. Cleaning SDH and Reflowing Dialogue (`subzero fix`)

Strips sound effects, speaker tags, HTML tags, ASS override codes, and normalizes smart quotes:

```console
# Clean a single subtitle file in place
subzero fix movie.srt

# Clean an entire movie or TV library, preserving originals in a backup folder
subzero fix /media/library --backup ~/subs-backup

# Preview changes without modifying files (dry run)
subzero fix show.srt --dry-run -v

# Keep musical symbols while stripping other sound effects
subzero fix concert.srt --keep-music
```

### 2. Checking timing and applying container delay

Install the optional speech dependencies for timing verification:

```console
pip install 'subzero-cli[sync]'
subzero verify-sync movie.mkv movie.pt-BR.srt
```

`verify-sync` emits JSON with per-window offset, correlation and confidence.
Exit codes are `0` for pass, `2` for rejection, `3` for inconclusive evidence and
`4` for an operational error. It never changes the subtitle. FFmpeg extracts audio
locally, and speech detection runs locally; no audio is uploaded. References are
cached under `~/.cache/subzero/references`, keyed by video identity and validator
version. Use `--cache DIR` to choose another location.

Verification examines overlapping windows across the dialogue span. The default
subtitle tolerance is 500 ms. Sparse speech, ambiguous correlations and isolated
mismatches remain inconclusive. Repeated timing mismatches reject the subtitle.
Embedded dialogue tracks are used only when local audio evidence supports them;
otherwise speech activity supplies the reference. Forced and commentary tracks
are excluded. Timing verification does not establish translation accuracy or
prove that every individual cue is correct. Videos from two minutes to four hours
are supported by the audio extraction path.

The Python API exposes `subzero.timing.correction` for conservative constant-delay
or linear framerate correction. Fit and held-out windows must agree, and consumers
must verify the corrected file again before accepting it. Different cuts and
irregular drift do not receive automatic piecewise repairs.

`sync` attempts speech-based correction and rechecks the result. When that is
unavailable, it falls back to container audio/video start-time skew. Its output
identifies the method used; a container-skew result alone does not verify dialogue
synchronization:

```console
# Try speech alignment, with container delay as a fallback
subzero sync movie.mkv movie.srt

# Output the aligned subtitle to a new file
subzero sync movie.mp4 movie.srt -o movie.synced.srt
```

### 3. Shifting Timestamps (`subzero shift`)

Offsets timecodes forward or backward with millisecond accuracy and zero-floor protection:

```console
# Delay subtitles by 1.5 seconds
subzero shift episode.srt --seconds +1.5

# Convert framerate timing (e.g. PAL 25fps to Film 23.976fps)
subzero shift movie.srt --from-fps 25 --to-fps 23.976

# Advance subtitles by 800 milliseconds across a whole folder
subzero shift ./subs --seconds -0.800 --backup ./backup
```

### 4. Format Conversion (`subzero convert`)

Converts subtitle text and timing between SRT, WebVTT, ASS, and SSA. Format-specific styling may be lost when converting to a simpler format:

```console
# Convert WebVTT to SubRip
subzero convert video.vtt --to srt

# Convert ASS with complex styling into clean SRT and strip SDH cues in one step
subzero convert anime.ass --to srt --fix

# Batch convert a directory to WebVTT for browser streaming
subzero convert ./library --to vtt --pattern "*.srt"
```

### 5. Extracting Soft Subtitles from Video Containers (`subzero extract`)

Extracts embedded text subtitle streams without memorizing ffmpeg stream mapping arguments:

```console
# Inspect available streams in a container
subzero streams movie.mkv

# Extract the default text track to a .srt file next to the video
subzero extract movie.mkv

# Extract all English and Portuguese subtitle tracks from a series folder and clean SDH
subzero extract /media/series --all --language eng por --format srt --fix
```

### 6. Translation (`subzero translate`)

Translates subtitles through Ollama or an explicitly selected OpenAI-compatible API while retaining the source cue timestamps:

```console
# Translate English subtitles to Brazilian Portuguese using local Ollama
subzero translate episode.srt --to pt-BR

# Translate using OpenAI or Groq / DeepSeek / OpenRouter
subzero translate movie.srt --to es --provider groq --api-key "$GROQ_API_KEY"

# Select the local Hy-MT2 model and endpoint explicitly
subzero translate movie.srt --to es --model subzero/hy-mt2:7b --url http://127.0.0.1:11434

# Supply character genders to a provider that uses the generic translation prompt
subzero translate episode.srt --to pt-BR --provider openai --cast "Ana:f,Rick:m"
```

`--cast` applies to the generic translation prompts. Hy-MT2 and TranslateGemma use their native prompt formats and do not consume this option. Selecting a remote provider or Ollama URL sends subtitle text to that endpoint.

### 7. Bilingual Subtitle Merge (`subzero merge`)

Merges two language tracks into a single bilingual subtitle file (ideal for language learners and dual-audio streaming):

```console
# Merge English and Portuguese subtitles
subzero merge movie.en.srt movie.pt.srt -o movie.dual.srt

# Highlight secondary language with custom color
subzero merge anime.jp.srt anime.en.srt -o anime.bilingual.srt --color "#ffff00"
```

### 8. OpenSubtitles MovieHash (`subzero moviehash`)

Calculates the 64-bit file hash used by OpenSubtitles to identify video releases:

```console
subzero moviehash movie.mkv
```

### 9. Background Media Server Daemon (`subzero watch`)

Runs as a lightweight daemon to automatically clean and prep newly arrived subtitles:

```console
subzero watch /media/library --interval 300 --backup ~/sub-backup
```

### 10. Subtitle Worker Daemon (`subzero worker`)

Runs the subtitle service for Jellyfin and YUCAST. It matches downloads, verifies dialogue timing against speech VAD, and translates verified embedded or external subtitles before falling back to local transcription. On macOS, verified English translation sources also pass through native caption recovery when `OCR_ENABLED=1`.

The worker validates candidate subtitles before installation:

* **Configurable target validation:** Validates against `ACCEPTED_LANGS` when set (e.g. `ACCEPTED_LANGS=pt-BR`). If unset or empty, all languages are accepted.
* **Zero SDH cues:** Any bracketed cues, parentheticals, speaker prefixes, or music symbols are sanitized.
* **Verified dialogue sync:** Existing subtitles with SDH cues are audited, sanitized, and updated in place once speech alignment is confirmed.
* **Layout:** Lines must fit within 42 characters. Collapsed dialogue turns (`- Person 1! - Person 2?`) are separated into distinct lines with standard `- ` prefixes.

```console
# Start the worker daemon
subzero worker serve

# Check worker health and queue status
subzero worker status

# Inspect active and recent queue jobs
subzero worker jobs

# Trigger a library sweep
subzero worker sweep

# Check subtitle coverage across the library
subzero worker coverage

# Inspect recent subtitle sync audits
subzero worker audits

# Summarize jobs waiting on review (sent to ntfy when NTFY_TOPIC is set)
subzero worker triage

# Open an interactive console for live monitoring and control
subzero worker console

# Request a graceful shutdown
subzero worker stop
```

The interactive console is a read-eval loop over the worker API. Type `help` for the command list. It covers health, the queue, job inspection, cancellation and resume, manual enqueueing, sweeps, coverage, audits, media streams, and OpenSubtitles search, plus a `watch` live refresh. It needs only the standard library, so it runs from any host that can reach the worker.

Set `NTFY_TOPIC` to receive phone pings. Manual jobs notify on every outcome; automatic jobs stay quiet on success and only ping on failure, review, or a quota pause. The daily auto sweep sends one low-priority summary with the enqueued mix, and `worker triage` pushes its digest on demand.

The worker binds to `127.0.0.1:8787`. Configure `JELLYFIN_URL`, `JELLYFIN_API_KEY`, and `BEARER_TOKEN` before starting it. API requests require `Authorization: Bearer TOKEN`.

`worker status` reports the transcription model and device (`whisperDevice`), translation provider and model, queue, and runner state. Its `gpu` field measures free NVIDIA VRAM and is `null` on Apple Silicon. For Ollama translation, use `ollama ps` to inspect where the model is running.

Configuration is read from an explicit `--env` file, the current directory's `.env`, the package/project `.env`, or `~/.config/subzero/.env`, in that order. Process environment variables override file values. The retired `~/.config/srtworker/.env` location is no longer discovered automatically; select it with `--env` during migration. The `srtworker` command and the existing macOS launchd label remain compatible.

Library sweeps use `AUTO_ENABLED`, `AUTO_WINDOW_START`, `AUTO_WINDOW_END`, and `WATCH_INTERVAL`. `IDLE_SHUTDOWN_MINUTES=15` enables idle shutdown; set it to `0` for a continuously running service.

#### qBittorrent completion trigger

The worker can watch qBittorrent and start a subtitle job the moment a torrent finishes downloading, without waiting for the auto sweep window. Enable it in the environment file:

```dotenv
QBT_ENABLED=1
QBT_URL=http://127.0.0.1:8585
QBT_API_KEY=qbt_your_api_key
QBT_CATEGORIES=movies,tv shows
QBT_POLL_INTERVAL=30
```

The worker polls `QBT_URL` every `QBT_POLL_INTERVAL` seconds using the `X-API-Key` header. When a torrent with `progress 1.0` in one of `QBT_CATEGORIES` appears, it refreshes the matching Jellyfin library, waits for the new item to be indexed, and enqueues an `audit` job for it in the first accepted language. Each torrent fires once; seen hashes are recorded under `SYNC_CACHE/qbt-seen.json`. A season pack enqueues one job per episode found under the download folder.

The mapping from download folder to Jellyfin library is derived at startup from `GET /Library/VirtualFolders`, so the qBittorrent category save path must fall inside a Jellyfin library location. These jobs are automatic: quiet on success, retried on failure, and claimed even when `AUTO_ENABLED=0`.

#### Configuring Target and Accepted Languages

`ACCEPTED_LANGS` restricts which target language tags the worker accepts. An empty value accepts all tags; language-specific content checks still apply where implemented.

Configure `ACCEPTED_LANGS` in your `.env` file:

```dotenv
# Restrict worker strictly to Brazilian Portuguese
ACCEPTED_LANGS=pt-BR

# Or accept multiple designated languages
ACCEPTED_LANGS=pt-BR,en,es

# Default: leave unset or empty to accept any valid subtitle language
ACCEPTED_LANGS=
```

Before transcribing audio that needs translation, the worker checks the selected translation provider. A translation failure leaves the job for review, without switching providers. DeepL Free quota holds are separate from translation failures. Malformed or incomplete output fails validation before installation. These checks do not certify the meaning of every translated sentence.

#### DeepL Free API

Select the Free API explicitly in the worker's environment file:

```dotenv
TRANSLATION_PROVIDER=deepl-free
```

Supply `DEEPL_API_KEY` through the process environment or a protected environment file. On macOS, an absent key is read from the Keychain entry with service `subzero.deepl.api-free` and account `worker`. The key is retrieved only when this provider is selected. It is excluded from configuration representations and health responses.

The provider accepts only Free keys ending in `:fx` and connects only to `https://api-free.deepl.com`. It sends English subtitle text to DeepL for Brazilian Portuguese translation. Local OCR and transcription remain separate stages. Selecting this provider does not start Ollama for translation or fall back to another translator.

The worker checks current account usage before admitting an episode for translation. If the remaining Free allowance cannot cover the episode, it pauses without submitting that episode's translation requests. Quota exhaustion during an admitted episode also pauses the work and preserves existing subtitles. Paid usage is never enabled automatically.

A paused job has `state: paused` and holds later queue claims. `/health` reports the paused count, and paused work remains included in the active queue count. After resolving the hold, send an authenticated `POST /jobs/{id}/resume` to requeue the same job; the worker checks admission again before translating. Resuming a job that is not paused returns HTTP 409. `DELETE /jobs/{id}` also accepts paused jobs for cancellation.

The quota journal at `SYNC_CACHE/deepl-free-usage.json` records conservative usage across worker restarts. A billing-period reset requires reconciliation against confirmed DeepL account usage before clearing that local floor. Restarting the worker or changing providers does not make a quota hold successful output.

#### Apple Foundation Models

Set `TRANSLATION_PROVIDER=applefm` to translate through Apple's on-device model. Start the local server with `fm serve` and leave it running; `APPLEFM_URL` defaults to `http://127.0.0.1:1976` and only loopback endpoints are accepted. The worker sends one sentence unit per request. A unit the model refuses or echoes falls back to LibreTranslate per unit when `TRANSLATION_FALLBACK=libretranslate` is set; without a fallback the block fails for review.

On-device generation is slower than the Argos CPU engine for long batches (measured on an M3 Pro: 47s versus 7s for 100 cues), so select `applefm` for on-device translation, not for throughput. `TRANSLATION_FALLBACK=applefm` is also accepted with `TRANSLATION_PROVIDER=deepl-free`; an episode that does not fit the Free allowance then continues on-device.

#### Native LibreTranslate CPU runtime

Set `TRANSLATION_PROVIDER=libretranslate` to use LibreTranslate's `argos-translate-lt` engine directly in an isolated Python process. This integration supports English to Brazilian Portuguese through the direct `en` to `pb` package, version 1.9. It does not run the LibreTranslate HTTP server or send dialogue to an external translation service.

The pinned installer currently supports native macOS 14+ ARM64 with CPython 3.13.15. Run it from a repository checkout with `uv` installed:

```sh
uv python install 3.13.15
python3 scripts/install_libretranslate.py \
  --python "$(uv python find 3.13.15)" \
  --runtime "$HOME/.local/share/subzero/libretranslate"
```

The installer verifies SHA-256 pins for all 17 wheels, the translation archive and the English MiniSBD model. It creates a separate environment, installs the wheels offline, checks their declared dependencies, and records the pins in `installation.json`. Installation does not load models or run translation. Use `--model /path/to/translate-en_pb-1_9.argosmodel` and `--segmenter /path/to/en.onnx` to reuse verified local model files. Existing runtime directories are refused; install a replacement into a new directory before changing the worker configuration.

Select the runtime in the worker's environment file:

```dotenv
TRANSLATION_PROVIDER=libretranslate
LIBRETRANSLATE_RUNTIME=/absolute/path/to/.local/share/subzero/libretranslate
```

The default path is `~/.local/share/subzero/libretranslate`. Keep its `config` directory empty. Runtime readiness checks verify the pinned package, segmenter and engine versions without loading a model. Translation runs in a short-lived CPU process with network connections disabled, one inter-op thread and two intra-op threads. It translates complete sentence units, then restores the original cue timestamps. It does not accept separate context or start Ollama. Engine and model settings are part of the translation cache identity, so changing providers does not reuse an Ollama translation.

The core worker dependencies remain separate. This runtime does not install Torch, spaCy, a container runtime or additional language packages. Preserve the bundled model notices; the English to Brazilian Portuguese package is licensed under CC BY 4.0.

#### Optional Ollama translation

`TRANSLATION_PROVIDER=ollama` remains the default for existing installations. Ollama retains the configured model between batches and releases it when the operation ends.

The worker accepts these Ollama controls:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Translation server endpoint. |
| `OLLAMA_MODEL` | `subzero/hy-mt2:7b` | Corrected local Hy-MT2 7B Q6_K package. |
| `OLLAMA_KEEP_ALIVE` | `2m` | Retain the model between subtitle batches. |
| `OLLAMA_NUM_CTX` | `4096` | Bound the context allocated for a batch. |
| `OLLAMA_NUM_PREDICT` | `2048` | Bound the generated response. |
| `OCR_ENABLED` | `1` on macOS, `0` elsewhere | Recover burned-in captions before translating verified English sources. |

For a 16 GB Apple Silicon machine, use CPU transcription and a bounded translation context:

```dotenv
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
OLLAMA_MODEL=subzero/hy-mt2:7b
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=2048
OLLAMA_KEEP_ALIVE=2m
```

Whisper transcribes the source language; the selected provider translates into the requested target language. Provision the configured Whisper model in the local Hugging Face cache before starting transcription. The worker requires a complete cached model and does not download it during a job. When `WHISPER_DEVICE` is unset, the worker picks `mlx` on Apple Silicon and `cuda` elsewhere. On Apple Silicon, `mlx` runs the same weights through mlx-whisper (the `mlx-community/whisper-` model must be the cached one), roughly 2.5x faster than CPU faster-whisper. The mlx path gates decoding with the same Silero VAD speech spans as the CPU path, so music and montage do not turn into invented dialogue.

On macOS, a shared compute policy serializes Ollama translation, Vision OCR, and Whisper across the CLI, worker, and HTTP transcription requests. Subzero stops the managed Ollama daemon and its runners before OCR or transcription, then starts it when Ollama translation is selected. Ollama translation retains ownership across its batches and fully stops the daemon afterward. Whisper runs in a separate process and exits before handoff, including when configured for CPU execution. The native LibreTranslate engine uses CPU execution and does not start the Ollama daemon.

Enable this behavior with `~/.config/subzero/compute.json`, or set `SUBZERO_COMPUTE_CONFIG` to an absolute policy path. All participating processes must use the same policy and lock. Without a policy, portable behavior is retained. An explicitly selected missing or invalid policy stops the operation.

| Policy field | Required value |
| --- | --- |
| `ollama_url` | Plain HTTP loopback endpoint matching the translation configuration. |
| `ollama_launch_agent` | Absolute path to a dedicated user launch-agent plist running only an absolute Ollama executable with `serve`. Its `OLLAMA_HOST` must match `ollama_url`. |
| `lock_path` | Shared absolute lock-file path in an existing directory. |
| `startup_timeout` | Optional service startup limit in seconds, default `30`. |
| `shutdown_timeout` | Optional shutdown limit in seconds, default `30`. |

Policy and launch-agent files must be regular files owned by the current user and not writable by others. Native children retain the shared lock until they exit, even if their parent process stops. If shutdown cannot be verified, further compute stays blocked. Commands and applications outside this policy do not participate in its lock; keep other GPU workloads paused during processing. A cancelled job can remain in cleanup while its native child exits, so the API's cancelled state alone does not prove resources have been released.

Hy-MT2 and `qwen3.5:9b` join bounded consecutive fragments from the same speaker, translate each sentence unit once, and distribute the generated words across the original cue timestamps. Batches keep these units together. Only Hy-MT2 uses prior context: complete units can receive up to 32 preceding source cues, capped at 6,000 characters, plus the programme title. The context excludes the current unit and later dialogue; unfinished fragments receive no background context. Qwen retains the generic JSON translation prompt without this context.

For Portuguese output, source units containing `Immunity Idol` or `Tribal Council` use [Tencent's terminology prompt](https://github.com/Tencent-Hunyuan/Hy-MT2#hy-mt2-translation-task-instruction-examples-chinese-english-comparison) with the Portuguese game terms, including plurals. The language guard rejects these terms if they remain in English.

TranslateGemma remains supported when explicitly selected. It receives its exact documented single-user prompt, including two blank lines before the source text, and requires known source and target languages. Validation rejects empty, truncated, malformed, or untranslated output before installation.

For TranslateGemma in the CLI, name an English input `episode.en.srt` or `episode.eng.srt` so the source language can be identified. Worker translation sources carry their language explicitly.

The default model is a local Hy-MT2 7B Q6_K package. The community 7B package needs a different prompt template from the smaller variants, and its EOS metadata incorrectly identifies `$` as an end token. Prepare a separate corrected copy before selecting it:

The preparation command requires POSIX file-open protections, available on macOS
and Linux. It stops on unsupported systems before opening the source or creating
files. The prepared model can then be imported into Ollama on another platform.

```sh
ollama pull kaelri/hy-mt2:7b
subzero_model_file=$(ollama show kaelri/hy-mt2:7b --modelfile | sed -n 's/^FROM //p')
subzero_model_stage="$HOME/.cache/subzero/models/hy-mt2-7b"
mkdir -p "$(dirname "$subzero_model_stage")"
python3 -m subzero.hymt2 "$subzero_model_file" "$subzero_model_stage"
ollama create subzero/hy-mt2:7b -f "$subzero_model_stage/Modelfile"
```

Run the preparation command with the Python environment where Subzero is installed. It requires a new staging directory, changes only the EOS metadata in the copy, and verifies the copied bytes. It retains the original model and tensor weights, and records hashes in `verification.json`. The template and end tokens follow [Tencent's 7B tokenizer](https://huggingface.co/tencent/Hy-MT2-7B/blob/main/chat_template.jinja).

For a dedicated Ollama server on a small machine, set `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_MAX_LOADED_MODELS=1` in the server environment. Keep it bound to loopback when the worker runs on the same host.

### 11. OpenSubtitles Contribution (`subzero contribute`)

Contributes verified subtitles back to the OpenSubtitles community catalogue:

```console
# Simulate contribution run without uploading
subzero contribute --dry-run --lang pt-BR --limit 20

# Upload verified subtitles with SQLite deduplication tracking
subzero contribute --env .env --lang pt-BR --limit 50
```

The contributor supports English and Brazilian Portuguese. Portuguese candidates pass the worker's SDH, layout, encoding, and language checks before upload. An SQLite database (`contributions.db`) tracks content hashes to prevent repeated submissions. `--dry-run` reads the catalogue and local files without uploading subtitles.
 

---
 
### 12. Native caption recovery (`subzero fill-gaps` / `subzero ocr-sync`)

 
Recovers burned-in open captions (e.g. whispered dialogue and challenge instructions) that were omitted from broadcast SDH tracks:
 
```console
# Recover and translate on-screen captions with Apple Vision and local Ollama
subzero fill-gaps movie.mkv movie.pt-BR.srt --to pt-BR
 
# Run recovery without modifying the subtitle file
subzero fill-gaps episode.mkv episode.pt-BR.srt --dry-run
```

`--dry-run` still performs frame extraction, OCR, and translation. Use `--to none` to inspect the recovered English captions without translation.

The CLI scans intervals without dialogue subtitles, including quiet passages without detected speech. Recovered cues are clipped to those intervals so they do not overlap existing dialogue.

Native Vision uses accurate English recognition and language correction. Caption position, size, horizontal angle, and enclosed white lettering filter out unrelated text. Accepted text boxes have their lower edge within the bottom 19% of the frame; recognition extends above that boundary to preserve the full letters. Dense lower-screen credit layouts and tall centered titles with stacked uppercase labels are excluded before individual rows are selected. Text boxes on the same physical row are read from left to right. Nearby frames and region retries help resolve text candidates and fluctuating boxes in single-line and two-line captions.

The worker can optionally retry unresolved whole-video caption intervals with a local image model after native Vision. Set both `OCR_RESCUE_MODEL` and `OCR_RESCUE_MODEL_DIGEST` to an installed vision-capable Ollama model and its exact 64-character SHA-256 digest from `/api/tags`. `OLLAMA_URL` must be a plain HTTP loopback endpoint. These settings are independent of the translation provider; native LibreTranslate translation does not use this model.

Caption rescue captures the exact sampled frames again and makes one image request for each uncached admitted frame. Native blank and title-card decisions remain authoritative. The model identity is checked before and after recognition; results, image crops and recognition evidence are cached under `SYNC_CACHE/caption-rescue`. Timing and caption checks still apply, and loss of a confirmed caption stops for review. This extra reading does not certify every word, short fragment or censorship marker. Inspect retained source candidates and evidence when quality remains uncertain.

The macOS worker exposes the same recovery as an explicit `recover_gaps` job through
`POST /jobs`, with `itemId` and `targetLang` (for example, `pt-BR`). It reads the
existing target sidecar, recovers captions with Apple Vision, and translates them
using the configured translation provider. The merged file must pass timing and subtitle
checks before installation. The worker backs up the original under
`SYNC_CACHE/backups` and preserves it when recovery, translation, or validation
fails. This job does not download subtitles or regenerate the episode from audio.
Regular audits do not run OCR; `SYNC_AUDIT_ONLY=1` prevents replacement.

When `OCR_ENABLED=1`, the worker also applies recovery automatically before translating a verified English source. It defaults to enabled on macOS and disabled elsewhere. This setting controls ordinary translation jobs; explicit `recover_gaps` and `repair` jobs request OCR themselves. An existing translation that passes its audit is kept. Submit a repair job to regenerate an existing poor translation.

Use `kind: "repair"` with the same endpoint when the existing translation needs
to be regenerated. Repair requires a verified English embedded track or English
sidecar. It scans the full supported video duration for burned-in English captions,
including captions shown while another person is speaking in the subtitle track.
The worker uses this same source scan for ordinary verified English translation
jobs when `OCR_ENABLED=1`.

Use `kind: "rebuild"` when no verified subtitle source is available. With `OCR_ENABLED=1`, English audio is transcribed from the same audio stream used by the independent timing reference. The transcript must pass timing checks before entering the full caption-recovery and translation pipeline. A detected-language mismatch on known English audio stops for review. Rebuild preserves the target snapshot taken before source selection or transcription, including when it finds an existing source sidecar. Failed retries repeat transcription; completed OCR and translation work can use their existing caches.

The source scan samples at two frames per second, then scans caption transitions at ten frames per second to refine their starts and ends. Timestamps come from the selected input frames. A one-nanosecond comparison tolerance prevents floating-point rounding from skipping a frame at the sampling boundary. Native recognition and frame sampling still limit timing precision; this does not establish word-level audio alignment.

If the dense scan loses a caption confirmed by repeated source frames, recovery stops for review and keeps the installed subtitle. OCR comparison preserves the position of underscore censorship bars, so marked and unmarked readings are not treated as equivalent.

Repeated dense observations can correct a coarse spelling error, while names, negation, and numbers remain protected. English contraction punctuation and common I/l errors in those contractions are normalized before timing verification. Brief missing lines are restored only between matching full observations; an initial one-line caption keeps its own timing. Rapid returns between near-identical word variants stop translation for review, including when an English source is reused from cache. This catches repeated OCR flicker, but does not certify every word or detect every two-reading ambiguity.

Completed scan windows are cached under `SYNC_CACHE/caption-scans`, including blank frames. Retries reuse their observed text and timestamps, then rerun caption and timing validation. Changes to the video, recognition version, native executable, macOS, FFmpeg, or scan settings invalidate the affected observations. Temporary video frames are discarded.

Each dialogue cue and recovered caption retains its timing through translation
and cleanup. Sentence-unit providers translate each unit once. When captions overlap, the
worker combines their translated text into successive display intervals with no
overlapping output cues. The final display cue count can therefore differ from
the source cue count. Speaker labels remain available during translation and are
removed from the delivered subtitle. The regenerated file must pass timing,
language, and dialogue-preservation checks before replacement.

For example, send this body to `POST /jobs` with the worker bearer token:

```json
{
  "itemId": "JELLYFIN_ITEM_ID",
  "kind": "repair",
  "targetLang": "pt-BR"
}
```

Inspect `GET /jobs/{id}` for progress and the final `outcome`, or use
`DELETE /jobs/{id}` to cancel. A job that needs review reports
`outcome: "needs_review"`; the compatibility `state` field reports `failed`.

Complete English and regenerated target subtitles are retained under
`SYNC_CACHE/candidates`. English OCR results are cached under
`SYNC_CACHE/ocr-sources`, keyed by video fingerprint, source content, and OCR
version. Translation retries reuse this source without rescanning the video.
Repair preserves the installed target when failure or cancellation occurs before
installation and rejects replacement if that target changed while the job was
running. An audit or media-server refresh error after installation can report a
failed job even though the subtitle was replaced; inspect the installed file and
job details before retrying. Repair does not fall back to downloads or audio
transcription.

New sidecars require filesystem support for atomic creation without replacing an existing file. If the media filesystem cannot provide it, the job ends in `needs_review` and retains the staged candidate.

Successful translation blocks are saved atomically under `SYNC_CACHE/translations`.
An interrupted repair resumes from these blocks after checking the exact source,
output integrity, timing, cue count, and target language. Changing the English
source, configured model, context or output limits, or translation prompt version
starts a new translation cache. Partial progress is never installed as a sidecar.

---

## Integration with Media Servers

### Sonarr custom script

Configure a Sonarr Custom Script hook for downloads:
```bash
#!/usr/bin/env bash
if [ "$sonarr_eventtype" = "Download" ]; then
    subzero extract "$sonarr_episodefile_path" --fix
    subzero fix "$(dirname "$sonarr_episodefile_path")" --pattern "*.srt"
fi
```

For Radarr, use `radarr_eventtype` and `radarr_moviefile_path` in the equivalent hook.

### Systemd Service (Linux Home Server)

Create `/etc/systemd/system/subzero-watch.service`:
```ini
[Unit]
Description=Subzero Subtitle Watcher Daemon
After=network.target

[Service]
Type=simple
User=media
ExecStart=/usr/local/bin/subzero watch /media/library --interval 300
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

The worker daemon runs the same way on Linux, pointing `ExecStart` at
`subzero worker serve` with a `--env` file that holds `JELLYFIN_URL`,
`JELLYFIN_API_KEY`, and `BEARER_TOKEN`.

### Windows and Linux notes

The core CLI, watcher, and worker are cross-platform and CI-tested on Linux, macOS, and Windows. A few features are deliberately gated to a platform:

* **Native caption recovery** (`fill-gaps`, `OCR_ENABLED=1`) uses Apple Vision and runs only on macOS. The worker defaults `OCR_ENABLED` to `1` on macOS and `0` elsewhere.
* **Shared compute policy** (`~/.config/subzero/compute.json`) coordinates Ollama, Vision, and Whisper through launchd and is macOS-only. Leave the policy unset to keep portable behavior on Linux and Windows.
* **DeepL Free keychain fallback** reads the key from macOS Keychain; on Linux and Windows set `DEEPL_API_KEY` explicitly.
* **Native LibreTranslate runtime** (`scripts/install_libretranslate.py`) currently supports macOS 14+ ARM64 only. On Linux and Windows select Ollama or a LibreTranslate/Argos install you run yourself.

`subzero worker start` uses launchd on macOS. On Linux and Windows there is no bundled service manager: run `subzero worker serve` in the foreground, or wrap it in systemd (Linux) or Task Scheduler/NSSM (Windows). All file paths, sidecar matching, and the interactive console handle Windows separators and loopback HTTP regardless of host.

---

## Supported Formats Matrix

| Kind | Extension | Supported Operations |
| :--- | :--- | :--- |
| **SubRip** | `.srt` | Read, Clean, Convert, Shift, Extract, Translate, Sync |
| **WebVTT** | `.vtt`, `.webvtt` | Read, Convert, Extract |
| **Advanced SubStation Alpha** | `.ass` | Read, Convert, Extract |
| **SubStation Alpha** | `.ssa` | Read, Convert, Extract |
| **Video Containers** | `.mkv`, `.mp4`, `.mov`, `.webm`, `.avi`, `.mlv`, `.ts`, `.m2ts` | Stream Inspection, Soft Subtitle Extraction, Audio Probe |
| **Character Encodings** | UTF-8, UTF-8-BOM, CP1252, Latin-1, ISO-8859-1 | Automatic detection and decoding to UTF-8 |

Convert other formats to SRT before using the subtitle cleanup, timing, or translation workflows.

---

## Options and Configuration

| Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| **Languages** | `--lang CODE...` | Language codes for role dictionary matching (`en`, `pt`, `es`, `fr`, `de`, `it`) | `en pt` |
| **Max Line Length** | `--max-line N` | Maximum characters per line before breaking dialogue | `42` |
| **Keep Brackets** | `--keep-brackets` | Keep bracketed cues such as `[Applause]` | `False` |
| **Keep Parens** | `--keep-parens` | Keep parenthetical cues such as `(Sighs)` | `False` |
| **Keep Music** | `--keep-music` | Keep musical symbols such as `♪` | `False` |
| **Keep Labels** | `--keep-labels` | Keep speaker labels such as `JOHN:` | `False` |
| **Rewrap All** | `--rewrap-all` | Re-break every cue, overriding intact human line breaks | `False` |
| **Backup Directory** | `--backup DIR` | Copy originals to a backup folder before writing changes | `None` |
| **Dry Run** | `--dry-run` | Report what would change without modifying files | `False` |

---

## Architecture and Engineering Decisions

1. **Human Line Breaks and Dialogue Turn Standards:**
   Professional release subtitles are timed and broken by human editors for reading pace. Subzero respects intact human line breaks while strictly enforcing quality gates: lines exceeding 42 characters are balanced near their natural midpoint, and collapsed dialogue turns sharing a line are split onto dedicated rows with standard hyphen prefixes.
2. **Conservative Label Matching:**
   ALL-CAPS text before a colon is treated as a speaker tag (`OFFICER:`). Lower-case words before a colon are matched against a closed dictionary of role words per language (`man:`, `mulher:`, `doctor:`, `medico:`) to avoid eating valid dialogue like `Score: 10`.
3. **Dialogue Dash Repair:**
   When an SDH cue containing one speaker is stripped from a two-speaker exchange, the remaining line has its leading dash cleaned to maintain dialogue integrity.

---

## Python Library API

Subzero can be integrated directly into Python pipelines:

```python
from subzero import (
    Options,
    fix_text,
    fix_file,
    convert_text,
    convert_file,
    shift_timestamps,
    shift_file,
    auto_sync_file,
    probe_audio_delay,
    moviehash,
    extract_from_video,
    list_subtitle_streams,
)

# Clean SDH from subtitle text in memory
result = fix_text(raw_srt_text, Options(max_line=40))
print(f"Cues: {result.cues}, Dropped: {result.dropped}, Rewrapped: {result.rewrapped}")

# Shift timecodes forward by 2.5 seconds
shifted_text, count = shift_timestamps(raw_srt_text, delta_seconds=+2.5)

# Convert between formats
vtt_result = convert_text(raw_srt_text, target="vtt", source="srt")

# Extract soft subtitles from video container
extract_from_video("movie.mkv", fmt="srt", languages=("eng",))
```

---

## Support and Sponsorship

If Subzero saved you time, improved your home media setup, or fixed your movie night, consider buying me a coffee:

<a href="https://buymeacoffee.com/yurirod"><img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=yurirod&button_colour=FFDD00&font_colour=000000&font_family=Inter&outline_colour=000000&coffee_colour=ffffff" /></a>

---

## Contributing

```console
git clone https://github.com/yuri-rod/subzero.git
cd subzero
pip install -e ".[dev]"
pytest
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history and version notes.

---

## License

MIT License (c) 2026 Yuri Barreira
