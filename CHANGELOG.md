# Changelog

All notable changes to Subzero are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
