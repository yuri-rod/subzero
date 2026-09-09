# Changelog

All notable changes to Subzero are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
