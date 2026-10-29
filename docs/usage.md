# Command and worker reference

For installation and a runnable example, start with the [README](../README.md).

## Interactive terminal menu

Run `subzero menu` to choose cleanup, extraction, conversion, timestamp shifting, and directory watching interactively. Use `subzero COMMAND --help` for the current options of each command.

---

## Commands and recipes

### Clean SDH and repair dialogue (`subzero fix`)

Remove sound effects, speaker tags, HTML tags, and ASS override codes, then normalize smart quotes:

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

### Inspect subtitle defects (`subzero check`)

```console
subzero check movie.srt
subzero check /media/library --pattern "*.srt"
```

The report counts SDH cues, collapsed dialogue turns, and long lines, and shows the
source encoding. It leaves files unchanged. Exit code `0` means no reported
issues; `1` means a file needs work, could not be read, or no files matched.

### Check timing and apply container delay

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

### Shift timestamps (`subzero shift`)

Move timecodes forward or backward in milliseconds. Timestamps that would become negative stop at zero:

```console
# Delay subtitles by 1.5 seconds
subzero shift episode.srt --seconds +1.5

# Convert framerate timing (e.g. PAL 25fps to Film 23.976fps)
subzero shift movie.srt --from-fps 25 --to-fps 23.976

# Advance subtitles by 800 milliseconds across a whole folder
subzero shift ./subs --seconds -0.800 --backup ./backup
```

### Convert formats (`subzero convert`)

Converts subtitle text and timing between SRT, WebVTT, ASS, and SSA. Format-specific styling may be lost when converting to a simpler format:

```console
# Convert WebVTT to SubRip
subzero convert video.vtt --to srt

# Convert ASS with complex styling into clean SRT and strip SDH cues in one step
subzero convert anime.ass --to srt --fix

# Batch convert a directory to WebVTT for browser streaming
subzero convert ./library --to vtt --pattern "*.srt"
```

### Extract embedded subtitles (`subzero extract`)

List subtitle streams or extract embedded text tracks with FFmpeg:

```console
# Inspect available streams in a container
subzero streams movie.mkv

# Extract the first matching text track to a .srt file next to the video
subzero extract movie.mkv

# Extract all English and Portuguese subtitle tracks from a series folder and clean SDH
subzero extract /media/series --all --language eng por --format srt --fix
```

### Translate subtitles (`subzero translate`)

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

### Benchmark translation (`subzero benchmark`)

Run the versioned `subtitle-dialogue-v1` workload of 12 synthetic English cues
against explicitly configured Ollama or OpenAI-compatible servers. The command
uses the installed translation clients, performs warmups, and writes JSON plus
a Markdown report beside it:

```console
subzero benchmark --config /absolute/path/to/benchmark.json --output /absolute/path/to/results.json
```

For example, configure a translator and an optional fixed Portuguese adequacy
judge in `benchmark.json`:

```json
{
  "repeats": 3,
  "warmup": 1,
  "batch_size": 4,
  "sample_interval": 0.25,
  "source_lang": "en",
  "target_lang": "pt-BR",
  "variants": [
    {
      "label": "candidate-q8",
      "provider": "openai",
      "url": "http://127.0.0.1:8080/v1",
      "model": "local-translator",
      "precision": "Q8_0",
      "runtime": "llama.cpp",
      "api_key_env": "BENCHMARK_API_KEY"
    }
  ],
  "judge": {
    "provider": "openai",
    "url": "http://127.0.0.1:8081/v1",
    "model": "fixed-judge",
    "api_key_env": "BENCHMARK_JUDGE_KEY"
  }
}
```

The values shown for repeats, warmup, batch size, sampling interval and languages
are the defaults. `variants` must contain at least one model. Each variant needs
`provider` (`ollama` or `openai`) and `model`; `label`, `precision`, `runtime`,
`url`, `api_key_env` and `model_file` are optional. An Ollama URL defaults to
`http://127.0.0.1:11434`; an OpenAI-compatible URL defaults to
`http://127.0.0.1:8080/v1`. API keys come from the named environment variable,
defaulting to `OPENAI_API_KEY`, and are omitted from requests when empty.
`model_file` adds a SHA-256 hash of a local weights file to the report.
Precision and runtime are supplied labels, so match them to the server actually
serving each model.

Provision and start the selected servers before the benchmark. It runs variants
serially and does not start external inference servers or quantize weights.
For separate model launches on shared hardware, run one variant per report with
the same workload, settings and fixed judge. Omit `judge` for translation and
deterministic checks only. The judge supports Portuguese targets and can use
either provider; the OpenAI-compatible judge requires JSON Schema structured
output support.

| Metric | Measurement |
| --- | --- |
| p50 and p95 latency | Complete workload translation time, including client retries and failed attempts. Warmup and judge execution are excluded. |
| Cues per second | Successfully translated cues divided by total measured translation time. |
| Tokens per second | Provider-reported completion tokens divided by translation time; unavailable when any measured run lacks token counts. |
| Guardrail pass rate | Fraction of attempted cues passing deterministic source/output checks and subtitle language/layout checks. |
| Mean adequacy and scored runs | Mean of available judge scores and the number of runs actually scored. Judge outages retain a missing score. |
| Memory, swap and VRAM | Sampled peak host memory use, peak swap use, and minimum free NVIDIA VRAM. These are host-wide observations; unsupported counters remain unavailable. |

The JSON also retains each output, error, resource sample, judge verdict, workload
hash, prompt version and runtime identity. Exit status `0` means every measured
translation completed. Quality rejections remain in the report and do not change
that exit status. See the [benchmark methodology and measured baselines](benchmarks/README.md)
for reproducible comparison instructions and results.

### Merge bilingual subtitles (`subzero merge`)

Combine two language tracks into one subtitle file:

```console
# Merge English and Portuguese subtitles
subzero merge movie.en.srt movie.pt.srt -o movie.dual.srt

# Highlight secondary language with custom color
subzero merge anime.jp.srt anime.en.srt -o anime.bilingual.srt --color "#ffff00"
```

### Calculate an OpenSubtitles hash (`subzero moviehash`)

Calculates the 64-bit file hash used by OpenSubtitles to identify video releases:

```console
subzero moviehash movie.mkv
```

### Watch a subtitle directory (`subzero watch`)

Poll a directory and clean new or changed subtitles automatically:

```console
subzero watch /media/library --interval 300 --backup ~/sub-backup
```

### Run the subtitle worker (`subzero worker`)

Runs the subtitle service for Jellyfin and YUCAST. It matches downloads, verifies dialogue timing against speech VAD, and translates verified embedded or external subtitles before falling back to local transcription. On macOS, verified English translation sources also pass through native caption recovery when `OCR_ENABLED=1`.

Verified workflows check target language, speech timing, SDH cleanup, and line
length before installing a candidate. `ACCEPTED_LANGS` restricts accepted target
tags; an empty value accepts all tags. Cleanup removes bracketed cues,
parentheticals, speaker prefixes, and music symbols, then separates collapsed
dialogue turns. The line-length limit is 42 characters.

Install the worker extra, copy [`.env.example`](../.env.example) to `.env`, and set
`JELLYFIN_URL`, `JELLYFIN_API_KEY`, and `BEARER_TOKEN`. Generate a bearer token with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`. Choose the translation
provider and provision its model or runtime before starting jobs.

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

The interactive console is a read-eval loop over the worker API. Type `help` for the command list. It covers health, the queue, job inspection, cancellation and resume, manual enqueueing, sweeps, coverage, audits, media streams, and OpenSubtitles search, plus a `watch` live refresh. It uses only the standard library and connects to the worker on loopback. For a remote worker, forward its port locally.

Set `NTFY_TOPIC` to receive phone pings. Manual jobs notify on every outcome; automatic jobs stay quiet on success and only ping on failure, review, or a quota pause. The daily auto sweep sends one low-priority summary with the enqueued mix, and `worker triage` pushes its digest on demand.

The worker binds to `127.0.0.1:8787`. Configure `JELLYFIN_URL`, `JELLYFIN_API_KEY`, and `BEARER_TOKEN` before starting it. API requests require `Authorization: Bearer TOKEN`.

`worker status` reports the transcription model and device (`whisperDevice`), translation provider and model, queue, and runner state. Its `gpu` field measures free NVIDIA VRAM and is `null` on Apple Silicon. `resourcePressure` contains the resource watchdog's enabled/blocked state, reason and latest snapshot. For Ollama translation, use `ollama ps` to inspect where the model is running.

Configuration is read from an explicit `--env` file, the current directory's `.env`, the package/project `.env`, or `~/.config/subzero/.env`, in that order. Process environment variables override file values. The retired `~/.config/srtworker/.env` location is no longer discovered automatically; select it with `--env` during migration. The `srtworker` command and the existing macOS launchd label remain compatible.

Library sweeps use `AUTO_ENABLED`, `AUTO_WINDOW_START`, `AUTO_WINDOW_END`, and `WATCH_INTERVAL`. `IDLE_SHUTDOWN_MINUTES=15` enables idle shutdown; set it to `0` for a continuously running service.

#### HTTP API

Send the bearer token with every request. For example, with `BEARER_TOKEN` set in
your shell:

```console
curl -H "Authorization: Bearer $BEARER_TOKEN" http://127.0.0.1:8787/health
```

| Request | Purpose |
| --- | --- |
| `GET /health` | Worker version, configured models, queue counts, runner status, and `resourcePressure`. |
| `GET /media/{item_id}` | Jellyfin item details, embedded stream indexes, and sidecars. |
| `POST /search` | Search OpenSubtitles with `itemId`, optional `query`, and optional `langs`. |
| `POST /jobs` | Queue a job with `itemId`, `kind`, `targetLang`, and optional `sourceId`. |
| `GET /jobs?limit=50` | Recent jobs and their outcomes. |
| `GET /jobs/{job_id}` | One job's progress, message, and result path. |
| `DELETE /jobs/{job_id}` | Request cancellation. |
| `POST /jobs/{job_id}/resume` | Requeue a paused job. Other states return HTTP 409. |
| `GET /coverage?lang=pt-BR` | List library items missing the requested language. |
| `GET /sync/audits` | The latest 100 timing audits and audit-only mode. |
| `POST /sweep` | Scan the library and enqueue missing work. |
| `POST /shutdown` | Request a graceful shutdown. |

A job's `outcome` is its full state. For compatibility, `state` reports `failed`
when `outcome` is `needs_review`. Read `message` and `resultPath` alongside the
state, especially after an installation or media-server refresh error.

Set `ASR_COMPAT=1` to expose authenticated multipart uploads at `POST /asr` and
`POST /detect-language`, using the `audio_file` field. `/asr` returns SRT by default
or plain text with `output=txt`. It transcribes in the detected source language;
the accepted `task` and `language` query parameters do not change that behavior.
Use a queued `whisper` job for transcription followed by translation.
`ASR_MAX_MB` limits upload size and defaults to 1024 MB.

#### qBittorrent completion trigger

The worker can poll qBittorrent for completed downloads and queue subtitle jobs outside the auto sweep window. Enable it in the environment file:

```dotenv
QBT_ENABLED=1
QBT_URL=http://127.0.0.1:8585
QBT_API_KEY=qbt_your_api_key
QBT_CATEGORIES=movies,tv shows
QBT_POLL_INTERVAL=30
```

The worker polls `QBT_URL` every `QBT_POLL_INTERVAL` seconds using the `X-API-Key` header. When a torrent with `progress 1.0` in one of `QBT_CATEGORIES` appears, it refreshes the matching Jellyfin library, waits for the new item to be indexed, and enqueues an `audit` job for it in the first accepted language. Each torrent fires once; seen hashes are recorded under `SYNC_CACHE/qbt-seen.json`. A season pack enqueues one job per episode found under the download folder.

The mapping from download folder to Jellyfin library is derived from `GET /Library/VirtualFolders`, so the qBittorrent category save path must fall inside a Jellyfin library location. These jobs are automatic: quiet on success, retried on failure, and claimed even when `AUTO_ENABLED=0`.

#### Target and accepted languages

`ACCEPTED_LANGS` limits target language tags in verified worker workflows. An empty value accepts all tags. Direct `translate` and `whisper` jobs do not enforce this setting at intake; configure their target language when submitting the job.

Configure `ACCEPTED_LANGS` in your `.env` file:

```dotenv
# Restrict worker strictly to Brazilian Portuguese
ACCEPTED_LANGS=pt-BR

# Or accept multiple designated languages
ACCEPTED_LANGS=pt-BR,en,es

# Default: leave unset or empty to accept any valid subtitle language
ACCEPTED_LANGS=
```

Translation requires the selected provider to be available. Fallbacks run only when you configure `TRANSLATION_FALLBACK`: Ollama can switch to LibreTranslate at admission, DeepL Free can switch to LibreTranslate or Apple Foundation Models at admission, and Apple Foundation Models can retry refused or echoed units through LibreTranslate. Malformed or incomplete output fails validation before installation. Review translated dialogue against the source for meaning.

#### OpenAI-compatible worker translation

Set `TRANSLATION_PROVIDER=openai` to translate queued jobs through an API serving
`/chat/completions`, including a local llama.cpp or vLLM endpoint:

```dotenv
TRANSLATION_PROVIDER=openai
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_MODEL=local-translator
OPENAI_API_KEY=
OPENAI_TIMEOUT=120
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL, including any `/v1` prefix. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model identifier sent to the server. |
| `OPENAI_API_KEY` | Empty | Optional bearer credential; leave empty for a local server without authentication. |
| `OPENAI_TIMEOUT` | `120` | Positive finite request timeout in seconds. |

Subzero validates the endpoint, model and timeout before translation and checks
response size, completion status, output format and cue count. The configured
server owns its model lifecycle. Changing the endpoint, model, prompt settings
or timeout changes the translation cache identity, preventing reuse across
different configurations. Credentials are excluded from that identity and health
responses. This provider has no automatic provider fallback.

Portuguese semantic QA still uses the separately configured Ollama judge below.
Set `SEMANTIC_QA=0` to disable that worker check when no judge is intended.

#### Translation quality checks

`SEMANTIC_QA=1`, the default, enables checks on full worker translations into Portuguese.
Deterministic checks flag extreme length changes, mistranslated game terms, and
repeated translations for distinct dialogue. An Ollama judge scores up to 40
source/output pairs in batches of 20, including suspect cues. Output is rejected when the mean
score falls below 3.5 out of 5, at least two sampled cues have critical defects,
or at least five cues trigger deterministic checks.

For a rejected result with at most 40 identified defective cues, the worker makes
one targeted repair pass through the selected translation provider. It translates
each flagged cue individually, using the existing malformed-output retry rules.
Flags include adequacy scores below 3.5, critical judgments and deterministic
defects. Unflagged text, cue indexes and timestamps remain unchanged.

Each repaired cue must pass structural and language validation. The worker then
checks the complete candidate again, keeping every repaired cue in the judge
sample, and requires all original defects to clear. If a scored rejection is
followed by a judge outage, the repair remains rejected. More than 40 defects,
missing actionable cue IDs, or a failed repair keep the existing failure or
review handling. There is no second semantic repair pass.

When a translation cache is in use, blocks containing confirmed defects are
invalidated so later jobs regenerate them. Other cached blocks remain reusable.
Cancellation is checked around each repair and before accepting the candidate;
an in-flight provider request may finish before cancellation takes effect.

The judge uses `OLLAMA_URL` and `OLLAMA_MODEL`, even when another provider performs
the translation. If the initial judge request is unavailable, deterministic checks still run.
Set `SEMANTIC_QA=0` to disable this extra check. Sampling does not establish the
accuracy of every cue; review representative dialogue yourself. The CLI and
standalone gap-recovery fragments do not use this worker check.

#### DeepL Free API

Select the Free API explicitly in the worker's environment file:

```dotenv
TRANSLATION_PROVIDER=deepl-free
```

Supply `DEEPL_API_KEY` through the process environment or a protected environment file. On macOS, an absent key is read from the Keychain entry with service `subzero.deepl.api-free` and account `worker`. The key is retrieved only when this provider is selected. It is excluded from configuration representations and health responses.

The provider accepts only Free keys ending in `:fx` and connects only to `https://api-free.deepl.com`. It sends English subtitle text to DeepL for Brazilian Portuguese translation. Local OCR, transcription, and the Ollama quality check are separate stages. Leave `TRANSLATION_FALLBACK` empty to keep translation on DeepL Free.

The worker checks current account usage before admitting an episode for translation. If the remaining Free allowance cannot cover it, the worker uses your configured fallback or pauses before submitting translation requests. Quota exhaustion during an admitted episode pauses the work and preserves existing subtitles. Paid usage is never enabled automatically.

A paused job has `state: paused` and holds later queue claims. `/health` reports the paused count, and paused work remains included in the active queue count. After resolving the hold, send an authenticated `POST /jobs/{id}/resume` to requeue the same job; the worker checks admission again before translating. Resuming a job that is not paused returns HTTP 409. `DELETE /jobs/{id}` also accepts paused jobs for cancellation.

The quota journal at `SYNC_CACHE/deepl-free-usage.json` records conservative usage across worker restarts. A billing-period reset requires reconciliation against confirmed DeepL account usage before clearing that local floor. Restarting the worker or changing providers does not make a quota hold successful output.

#### Apple Foundation Models

Set `TRANSLATION_PROVIDER=applefm` to translate through Apple's on-device model. Start the local server with `fm serve` and leave it running; `APPLEFM_URL` defaults to `http://127.0.0.1:1976` and only loopback endpoints are accepted. The worker sends one sentence unit per request. A unit the model refuses or echoes falls back to LibreTranslate per unit when `TRANSLATION_FALLBACK=libretranslate` is set; without a fallback the block fails for review.

Compare provider output and elapsed time on your own subtitle workload before selecting a default. `TRANSLATION_FALLBACK=applefm` is also accepted with `TRANSLATION_PROVIDER=deepl-free`; an episode that does not fit the Free allowance then continues on-device.

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

`TRANSLATION_PROVIDER=ollama` is the worker default. Install the configured model before queueing translation. Ollama retains it between batches and releases it when the operation ends.

The worker accepts these Ollama controls:

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Translation server endpoint. |
| `OLLAMA_MODEL` | `qwen3.5:9b-mlx` | Worker translation model. This default needs an Ollama server with MLX support; select an installed model suitable for your server. |
| `OLLAMA_KEEP_ALIVE` | `2m` | Retain the model between subtitle batches. |
| `OLLAMA_NUM_CTX` | `4096` | Bound the context allocated for a batch. |
| `OLLAMA_JOBS` | `1` | Parallel subtitle blocks per batch. Applies to generic Ollama models only; context-chained native models stay serial. Match the server `OLLAMA_NUM_PARALLEL`. |
| `TRANSLATION_FALLBACK` | _(empty)_ | With `libretranslate`, whole jobs route to the Argos engine when Ollama is down or the model is missing. The switch happens once at admission, never silently mid-batch. |
| `OLLAMA_NUM_PREDICT` | `2048` | Bound the generated response. |
| `OCR_ENABLED` | `1` on macOS, `0` elsewhere | Recover burned-in captions before translating verified English sources. |

To keep transcription on CPU while using a bounded translation context on Apple Silicon:

```dotenv
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
OLLAMA_MODEL=qwen3.5:9b-mlx
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=2048
OLLAMA_KEEP_ALIVE=2m
```

Whisper transcribes the source language; the selected provider translates into the requested target language. Provision the configured Whisper model in the local Hugging Face cache before starting transcription. The worker requires a complete cached model and does not download it during a job. When `WHISPER_DEVICE` is unset, the worker picks `mlx` on Apple Silicon and `cuda` elsewhere. On Apple Silicon, `mlx` runs the same weights through mlx-whisper (the `mlx-community/whisper-` model must be the cached one). The mlx path limits decoding to the same Silero VAD speech spans as the CPU path to reduce hallucinations during music and montage.

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

The CLI defaults to `subzero/hy-mt2:7b`, independently of the worker's `OLLAMA_MODEL`. To use the community Hy-MT2 7B Q6_K package, prepare a corrected copy: its prompt template differs from the smaller variants, and its EOS metadata incorrectly identifies `$` as an end token.

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

#### Resource pressure and deferred jobs

The resource watchdog is opt-in. Set one or more byte thresholds in the worker
environment to hold queued work when resources are unavailable:

| Setting | Default | Behavior |
| --- | --- | --- |
| `COMPUTE_MIN_MEMORY_AVAILABLE_BYTES` | Unset | Block when available host memory is below this value. |
| `COMPUTE_MAX_SWAP_USED_BYTES` | Unset | Block when used swap exceeds this value. |
| `COMPUTE_MIN_VRAM_FREE_BYTES` | Unset | Block when the least free NVIDIA GPU memory is below this value. |
| `COMPUTE_PRESSURE_INTERVAL` | `5` | Positive finite sampling interval in seconds. |
| `COMPUTE_PRESSURE_RETRY_SECONDS` | `30` | Positive finite delay before a deferred job becomes eligible again. |

Thresholds are nonnegative integer byte counts. For example,
`COMPUTE_MIN_MEMORY_AVAILABLE_BYTES=4294967296` requires at least 4 GiB available.
Host memory and swap measurements support macOS and Linux. Free VRAM requires
`nvidia-smi`; Apple Silicon does not report a separate NVIDIA VRAM value. If an
enabled threshold cannot be measured, admission remains blocked. With every
threshold unset, the watchdog does not sample or unload services.

The worker checks pressure before dispatch and at new local compute phases.
An affected manual or automatic job returns to `queued` with
`phase: "resource_pressure"` and a scheduled retry delay. This does not consume a
handler retry, including when the job has already reached its exception retry
limit. Once resources recover and the delay expires, normal queue selection
resumes. Cancellation retains priority over deferral.

The watchdog samples in the background and exposes `enabled`, `blocked`,
`reason` and `snapshot` through `resourcePressure` in `GET /health` and
`subzero worker status`. Snapshot counters use bytes and include a UTC timestamp.
Under the managed compute policy, pressure may stop an idle owned Ollama daemon
after acquiring its shared lease. Active compute ownership and native child
leases prevent that cleanup; unmanaged servers are not stopped. Running native
work finishes its current phase before later admission checks can defer the job.

### Contribute to OpenSubtitles (`subzero contribute`)

Contributes verified subtitles back to the OpenSubtitles community catalogue:

```console
# Simulate contribution run without uploading
subzero contribute --dry-run --lang pt-BR --limit 20

# Upload verified subtitles with SQLite deduplication tracking
subzero contribute --env .env --lang pt-BR --limit 50
```

The contributor supports English and Brazilian Portuguese. Portuguese candidates pass the worker's SDH, layout, encoding, and language checks before upload. An SQLite database (`contributions.db`) tracks content hashes to prevent repeated submissions. `--dry-run` reads the catalogue and local files without uploading subtitles.


### Recover burned-in captions (`subzero fill-gaps` / `subzero ocr-sync`)

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

Caption rescue captures sampled frames again, reuses cached results, and calls the configured image model for unresolved readings. Native blank and title-card decisions remain authoritative. The model identity is checked before and after recognition; results, image crops and recognition evidence are cached under `SYNC_CACHE/caption-rescue`. Timing and caption checks still apply, and loss of a confirmed caption stops for review. Inspect retained source candidates and evidence when readings conflict.

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

Repeated dense observations can correct a coarse spelling error, while names, negation, and numbers remain protected. English contraction punctuation and common I/l errors in those contractions are normalized before timing verification. Brief missing lines are restored only between matching full observations; an initial one-line caption keeps its own timing. Rapid returns between near-identical word variants stop translation for review, including when an English source is reused from cache. Two-reading ambiguities can remain unresolved.

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

When the embedded track is already trusted text, `kind: "translate"` with
`sourceId: "embedded:N"` (the Jellyfin stream index from `GET /media/{id}`)
skips the reference build, caption recovery, and audit ladder: it extracts the
track, translates it, and delivers the sidecar. Use this path for
trusted sources; use `embedded_translate` when the source needs verification
against the audio. A `translate` job from a sidecar uses the sidecar tag as
`sourceId` instead (for example, `"en"`).

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

## Media server integration

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

### systemd service on Linux

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

The core CLI, watcher, and worker are cross-platform and CI-tested on Linux, macOS, and Windows. These features have platform requirements:

* Native caption recovery (`fill-gaps`, `OCR_ENABLED=1`) uses Apple Vision and runs only on macOS. The worker defaults `OCR_ENABLED` to `1` on macOS and `0` elsewhere.
* The shared compute policy (`~/.config/subzero/compute.json`) coordinates Ollama, Vision, and Whisper through launchd and is macOS-only. Leave the policy unset to keep portable behavior on Linux and Windows.
* DeepL Free keychain fallback reads the key from macOS Keychain; on Linux and Windows set `DEEPL_API_KEY` explicitly.
* The native LibreTranslate installer (`scripts/install_libretranslate.py`) supports macOS 14+ ARM64 only. On Linux and Windows, select another supported provider. The worker does not connect to a LibreTranslate HTTP server.

`subzero worker start` uses launchd on macOS. On Linux and Windows there is no bundled service manager: run `subzero worker serve` in the foreground, or wrap it in systemd (Linux) or Task Scheduler/NSSM (Windows). All file paths, sidecar matching, and the interactive console handle Windows separators and loopback HTTP regardless of host.

---

## Supported formats

| Kind | Extension | Supported operations |
| :--- | :--- | :--- |
| SubRip | `.srt` | Read, Clean, Convert, Shift, Extract, Translate, Sync |
| WebVTT | `.vtt`, `.webvtt` | Read, Convert, Extract |
| Advanced SubStation Alpha | `.ass` | Read, Convert, Extract |
| SubStation Alpha | `.ssa` | Read, Convert, Extract |
| Video containers | `.mkv`, `.mp4`, `.mov`, `.webm`, `.avi`, `.mlv`, `.ts`, `.m2ts` | Stream Inspection, Soft Subtitle Extraction, Audio Probe |
| Character encodings | UTF-8, UTF-8-BOM, UTF-16 with a byte-order mark, CP1252, Latin-1 | Automatic detection and decoding to UTF-8 |

Convert other formats to SRT before using the subtitle cleanup, timing, or translation workflows.

---

## Cleanup options

| Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| Languages | `--lang CODE...` | Language codes for role dictionary matching (`en`, `pt`, `es`, `fr`, `de`, `it`) | `en pt` |
| Max Line length | `--max-line N` | Maximum characters per line before breaking dialogue | `42` |
| Keep brackets | `--keep-brackets` | Keep bracketed cues such as `[Applause]` | `False` |
| Keep parens | `--keep-parens` | Keep parenthetical cues such as `(Sighs)` | `False` |
| Keep music | `--keep-music` | Keep musical symbols such as `♪` | `False` |
| Keep labels | `--keep-labels` | Keep speaker labels such as `JOHN:` | `False` |
| Rewrap all | `--rewrap-all` | Re-break every cue, overriding intact human line breaks | `False` |
| Backup directory | `--backup DIR` | Copy originals to a backup folder before writing changes | `None` |
| Dry run | `--dry-run` | Report what would change without modifying files | `False` |

---

## Cleanup behavior

Subzero keeps intact line breaks unless a line exceeds the configured limit or
you pass `--rewrap-all`. Long lines are balanced near their natural midpoint;
collapsed dialogue turns get separate rows with `- ` prefixes.

ALL-CAPS text before a colon is treated as a speaker tag (`OFFICER:`). Lowercase
labels must match the selected language's role dictionary (`man:`, `mulher:`,
`doctor:`, `medico:`), so ordinary dialogue such as `Score: 10` survives cleanup.

When removing an SDH cue leaves only one speaker in an exchange, cleanup removes
the remaining line's dialogue dash.

---

## Python library API

Run this from a checkout to clean, shift, and convert the bundled example in memory:

```python
from pathlib import Path

from subzero import Options, convert_text, fix_text, shift_timestamps

source = Path("examples/dialogue.srt").read_text(encoding="utf-8")
cleaned = fix_text(source, Options(max_line=40))
shifted, _ = shift_timestamps(cleaned.text, delta_seconds=2.5)
webvtt = convert_text(shifted, target="vtt", source="srt")
print(webvtt.text, end="")
```

To extract English subtitles from your own video, with FFmpeg installed:

```python
from subzero import extract_from_video

extract_from_video("movie.mkv", fmt="srt", languages=("eng",))
```
