# Changelog

All notable changes to Subzero are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.10.2] - 2026-09-12

### Fixed

- Translate sentence continuations once and distribute the generated text across the original timestamps. Keep current and future dialogue out of background context to prevent repeated content between cues.
- Retain single-line and two-line captions when Vision's bounding boxes fluctuate between frames, confirm candidate text against nearby frames, and reject rotated prop text.
- Include native caption recovery in verified English translation jobs on macOS, controlled by `OCR_ENABLED`.
- Scan the full supported video duration during English source regeneration so burned-in captions are recovered even when they overlap existing dialogue. Translate source cues once and combine simultaneous dialogue into output intervals without overlaps; CLI gap filling remains limited to subtitle gaps.
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
