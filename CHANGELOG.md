# Changelog

All notable changes to Subzero are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Keep caption rescue clustering from merging distinct captions. Only consecutive frames where Apple Vision reads the same words with the same censorship marks share one model call, so caption transitions, name and number changes, and dropped censor bars always get their own reading. Invalidate older rescue caches that may hold propagated readings.
- Keep high-confidence Apple Vision readings as a direct pass-through without a model call. Verified identical to the model on every sampled cluster.
- Drop the parallel model dispatch and the compact-model prompt path. Measured against the local model they added no speedup and lowered reading accuracy, so rescue stays sequential on the configured model.

## [1.13.0] - 2026-09-13

### Added

- Add an explicit DeepL Free worker provider for English to Brazilian Portuguese. Check the character budget for the entire uncached episode before translation, preserve completed blocks, and pause the queue when the episode does not fit.
- Persist conservative character reservations across restarts and ambiguous requests. Keep the Free endpoint fixed, use environment or Keychain credentials, and require confirmed quota reconciliation after a billing reset.
- Expose paused jobs in worker health and add an authenticated resume endpoint. Resuming repeats episode admission before sending more text.

## [1.12.1] - 2026-09-13

### Added

- Add `subzero worker jobs`, `sweep`, `coverage`, and `audits` CLI commands to inspect daemon state, library coverage, recent audits, and trigger sweeps.
- Route worker execution cleanly between `SyncFlow` workflows and core `embedded`, `opensubtitles`, `whisper`, and `translate` handlers.

### Fixed

- Remove sound descriptions spanning multiple lines before forming translation units, so sound-only cues cannot consume dialogue timing anchors.
- Reconcile cropped fragments of a single caption row without duplicating dialogue or dropping an observed censorship mark. Invalidate older processed OCR scans and source caches.
- Use the same subtitle-region admission checks for coarse and dense frame replay. Keep conflicting evidence rejected and report its exact timestamp.


## [1.12.0] - 2026-09-13

### Added

- Add an optional native LibreTranslate worker provider using `argos-translate-lt` 1.12.1 and the direct English to Brazilian Portuguese package 1.9. Translation runs in an isolated CPU process without external translation requests or automatic provider fallback.
- Add a pinned runtime installer for macOS 14+ ARM64 and CPython 3.13.15, with verified model and dependency hashes, offline wheel installation, dependency checks, and a separate environment. Include the installer in source distributions.
- Add optional caption rescue through an installed local image model pinned by SHA-256 digest. Retry unresolved native Vision intervals using the exact sampled frames, preserve native blank and title-card decisions, and retain image and recognition evidence for review.

### Changed

- Keep complete sentence units together when translating with `qwen3.5:9b`; retain its existing JSON prompt without Hy-MT2 context.
- Check the selected worker translation provider before transcription and separate translation caches by provider, model and settings. Existing installations retain Ollama as their default provider.

### Fixed

- Detect rapid returns between short-word, initial and positioned censor-mark variants. Route unresolved readings through optional caption rescue or stop for review without rewriting the source cues.

## [1.11.2] - 2026-09-12

### Fixed

- Preserve the position of observed underscore censor marks when comparing native OCR candidates, region retries, and dense timing observations. Bar width and surrounding spacing remain equivalent; losing a confirmed mark during dense verification stops recovery for review.
- Keep censor marks out of word-count thresholds so short captions retain their existing correction limits.
- Report an unsupported platform clearly when a shared compute policy is selected without POSIX support. Ordinary operation without that policy remains portable.
- Close temporary subtitle and cache files before atomic installation on Windows, preserve existing line endings, and decode FFmpeg progress and diagnostics as UTF-8.
- Report the POSIX file-open requirement before inspecting or preparing a corrected Hy-MT2 model on an unsupported platform.

## [1.11.1] - 2026-09-12

### Fixed

- Prevent floating-point rounding from skipping OCR frames at exact sampling boundaries. Invalidate older scan observations so retries use the corrected cadence.

## [1.11.0] - 2026-09-12

### Added

- Route English audio rebuilds through transcription, independent audio timing checks, full caption recovery, and the existing translation pipeline when `OCR_ENABLED=1`.

### Fixed

- Extract the same audio stream used by the timing reference, validate its stream index, and stop for review when known English audio is detected as another language.
- Preserve the target snapshot taken before transcription or source selection across every rebuild fallback. Keep existing subtitles when validation fails or the target changes during processing.
- Match sidecars by literal video filename so bracketed release names are handled correctly and neighboring video names are preserved during cleanup.

## [1.10.9] - 2026-09-12

### Fixed

- Wait for launchd startup transitions to settle before validating the managed Ollama process and listener.

## [1.10.8] - 2026-09-12

### Added

- Add an optional shared macOS compute policy that serializes Ollama translation, Apple Vision OCR, and Whisper transcription across the CLI, worker, and HTTP transcription requests.

### Fixed

- Under the compute policy, stop Ollama and its runners before OCR or transcription, retain ownership across translation batches, and require child processes to exit before handing off resources.
- Keep the shared lock held by native children if their parent exits. Reject unsafe policy files and stop further compute when shutdown cannot be verified.
- Continue cancellation checks while FFmpeg or Whisper is silent or has closed its output stream. Propagate stream read errors and reject malformed transcription messages.

## [1.10.7] - 2026-09-12

### Fixed

- Exclude tall centered titles with stacked uppercase labels from caption recovery while preserving uppercase dialogue outside that layout.

## [1.10.6] - 2026-09-12

### Fixed

- Stop repeated unstable caption readings before translation, including when reusing an English source from cache.
- Normalize English contraction punctuation and common I/l recognition errors before timing validation.
- Tighten caption row placement checks to exclude higher credit text while retaining the full height of admitted letters.

## [1.10.5] - 2026-09-12

### Fixed

- Reject missing or unreadable explicitly selected environment files and `--env` options without a path.
- Recheck brief missing caption lines after confirmed spelling corrections, retaining the existing protections for names, negation, and numbers.

## [1.10.4] - 2026-09-12

### Fixed

- Use caption geometry, enclosed white lettering, and credit-layout checks to filter unrelated screen text. Validate native region metadata before selecting rows.
- Read text boxes on the same physical row from left to right and recover fragmented rows from matching region observations.
- Require repeated dense observations for spelling corrections. Preserve initial one-line captions, real text changes, and ambiguous native readings instead of replacing them with older coarse text.

## [1.10.3] - 2026-09-12

### Fixed

- Prevent region OCR retries from duplicating fragments already contained in a complete native caption line.
- Cache completed native scan windows so interrupted scans and validation retries can reuse observations without bypassing quality checks.

## [1.10.2] - 2026-09-12

### Fixed

- Translate sentence continuations once and distribute the generated text across the original timestamps. Keep current and future dialogue out of background context to prevent repeated content between cues.
- Retain single-line and two-line captions when Vision's bounding boxes fluctuate between frames, confirm candidate text against nearby frames, and reject rotated prop text.
- Include native caption recovery in verified English translation jobs on macOS, controlled by `OCR_ENABLED`.
- Scan the full supported video duration during English source regeneration so burned-in captions are recovered even when they overlap existing dialogue. Translate source cues once and combine simultaneous dialogue into output intervals without overlaps; CLI gap filling remains limited to subtitle gaps.
- Refine recovered caption boundaries with dense native frame samples around text transitions.
- Use Hy-MT2 terminology prompts for Portuguese game terms and reject English `Immunity Idol` and `Tribal Council` terms left in Portuguese dialogue.
- Remove automatic discovery of the retired worker configuration directory while retaining explicit `--env` selection and CLI compatibility.
- Correct CLI examples, model configuration, dependency requirements, and format support in the README.
- Set the package minimum to Python 3.10 and align CI with the worker's runtime requirements.

## [1.10.1] - 2026-09-12

### Fixed

- Recover burned-in captions across subtitle gaps, including whispers missed by voice detection, with native Apple Vision bundled in the package.
- Preserve source timing during OCR recovery and subtitle regeneration. Reject incomplete translations before replacing the installed file.
- Use the documented TranslateGemma prompt and the correct Hy-MT2 7B token format.
- Resume interrupted repair translations from validated blocks.
- Keep Whisper model loading offline, clean temporary audio after errors, and clamp transcription timestamps to the audio duration.

## [1.10.0] - 2026-09-12

### Added

- Speech gap analysis and native Apple Vision OCR caption recovery (`subzero fill-gaps` / `subzero ocr-sync`): audits audio speech intervals against subtitle timing, extracts frames across gaps, and recovers burned-in dialogue omitted from broadcast SDH tracks.
- Dynamic half-split fallback for Ollama and TranslateGemma translation: automatically bisects cue batches on line-count mismatches to prevent progressive cue drift.
- Seek before FFmpeg input decoding and limit extraction to the selected speech gaps.

### Fixed

- Lone surrogate character handling: safely handles unpaired unicode surrogates from LLM outputs using replacement encoding during file writes.
- Markdown code block extraction: reliably parses formatted code blocks returned by local LLMs in subtitle translation.

## [1.9.0] - 2026-09-12

### Added

- Language completeness guard: checks that translations (e.g. pt-BR) remain consistent through the end of the file, rejecting incomplete community subtitles where text reverts to English.
- Redundant sidecar pruning: removes bare `.srt` files and intermediate `.en.srt` sidecars matching container embedded tracks when installing subtitles.
- Tag and timing guards: catches unclosed formatting tags, machine translation casing artifacts, and severe timing drift during excellence checks.

### Fixed

- Collapsed dialogue false positives: refined `is_collapsed()`, `PUNCT_DASH`, and `SPLIT_DIALOGUE` in `core.py` so single-speaker parentheticals, pauses, and ranges are not wrongly split or flagged.
- Multi-line dialogue formatting in translation: kept line breaks in `_parse_lines` during structured LLM responses instead of collapsing them with spaces.
- Target language propagation: forwarded `job.target_lang` and `accepted_langs` to all `sanitize_to_excellence` calls in `syncflow.py`.
- Rebuild stage failure reporting: corrected inverted failure reason in `_rebuild` validation.

## [1.8.0] - 2026-09-10

### Added

- Year and frame-rate match verification for subtitle candidates, with score promotion for same-group reencodes.
- IMDb ID normalization (no `tt` prefix or leading zeros) and logout after contribution runs.

## [1.7.0] - 2026-09-10

### Added

- OpenSubtitles match verification: feature-type agreement in title checks and strict hash-only search mode.
- Forced-track tracking: `foreign_parts_only` captured on candidates, surfaced in `/search`, deprioritized in automated picks.
- Broken-subtitle memory: empty downloads recorded in the attempts ledger and skipped by refetch and watch picks.
- Quota pre-check: download reset time tracked, automated refetch jumps to local resync on empty quota, manual jobs fail fast.
- pt-BR automated disclaimer on machine-translation uploads.

## [1.6.0] - 2026-09-10

### Added

- Character gender map for translation (`subzero translate --cast "Ana:f,Rick:m"`):
  - Resolves pronouns and adjective agreement against known speaker genders.
  - Neutral-first guidance in Ollama, TranslateGemma, and OpenAI-compatible prompts: prefers gender-invariable phrasing when the speaker is unknown and never guesses from stereotypes.

### Fixed

- `subzero extract` crash (`extract_from_video() got an unexpected keyword argument 'indices'`): CLI now passes the correct `indexes` parameter.

## [1.5.0] - 2026-09-08

### Added

- **Mandatory Line Break & Dialogue Turn Acceptance Rules:**
  - Automated detection and separation of merged or collapsed dialogue lines (e.g. `- Person 1! - Person 2?` and `Person 1! - Person 2?`).
  - Dialogue turns are split onto separate lines with standardized `- ` prefixes.
  - Safe boundary detection: distinguishes true speaker dialogue changes from parenthetical dashes and mid-sentence asides (`The plan - such as it was - failed`).
  - Multi-line cue inspection in `subzero check`: checks every row within every cue for collapsed turns or lines exceeding limits.
  - Strict line length balancing in `core.py`: ensures long runs are recursively balanced at natural punctuation and conjunctions to stay under 42 characters.
  - Subtitle translation pipeline (`subzero translate`) automatically enforces line breaking and dialogue splitting guards before saving output files.
  - Excellence guards reject any subtitle containing collapsed dialogues or lines longer than presentation limits.
- **Configurable Language Acceptance (`ACCEPTED_LANGS`):**
  - Worker daemon and excellence guards no longer enforce a hardcoded language constraint globally.
  - Language filtering is user-configurable via the `ACCEPTED_LANGS` environment variable (e.g. `ACCEPTED_LANGS=pt-BR` or `ACCEPTED_LANGS=pt-BR,en`). When unset or empty, Subzero accepts all valid languages worldwide.
  - Added documentation and examples for configuring preferred language filters in `.env`.

### Fixed

- Fixed edge case in `_tag_split` where multiple disjoint HTML tags (e.g. `<i>A</i> - <i>B</i>`) were incorrectly treated as a single outer wrapper.
- Fixed stray leading colon cleanup in `strip_label` when speaker tags had partial punctuation remnants.

## [1.4.0] - 2026-09-07

### Added

- OpenSubtitles contribution pipeline (`subzero contribute`) with SQLite deduplication tracking and dry-run support.
- Worker CLI commands for contribution and ledger maintenance.

## [1.3.0] - 2026-09-07

### Added

- Speech dialogue synchronization (`subzero verify-sync`) using local audio extraction and VAD verification.
- Constant offset delay repair and correlation metrics.

## [1.2.0] - 2026-09-06

### Added

- Warm model retention for Ollama LLM translation sessions.
- Incomplete translation detection and smaller batch retries.

## [1.0.0] - 2026-08-30

### Added

- Initial public release of Subzero: SDH removal, timing shift, format conversion, video track extraction, and interactive terminal menu.
