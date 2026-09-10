# Changelog

Release notes follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-09-10

### Added
- Character gender map for translation (`subzero translate --cast "Ana:f,Rick:m"`):
  - Instructs the model to use supplied speaker genders for pronouns and adjective agreement.
  - Prompts that support cast information request gender-neutral phrasing when the speaker is unknown.

### Fixed
- `subzero extract` crash (`extract_from_video() got an unexpected keyword argument 'indices'`): CLI now passes the correct `indexes` parameter.

## [1.5.0] - 2026-09-08

### Added
- Dialogue layout checks:
  - Automated detection and separation of merged or collapsed dialogue lines (e.g. `- Person 1! - Person 2?` and `Person 1! - Person 2?`).
  - Dialogue turns are split onto separate lines with standardized `- ` prefixes.
  - Safe boundary detection: distinguishes true speaker dialogue changes from parenthetical dashes and mid-sentence asides (`The plan - such as it was - failed`).
  - Multi-line cue inspection in `subzero check`: checks every row within every cue for collapsed turns or lines exceeding limits.
  - Balance long lines near punctuation and conjunctions in `core.py`, using the configured line-length limit.
  - Subtitle translation pipeline (`subzero translate`) automatically enforces line breaking and dialogue splitting guards before saving output files.
  - Subtitle quality checks reject any subtitle containing collapsed dialogues or lines longer than presentation limits.
- Configurable target languages (`ACCEPTED_LANGS`):
  - The worker and subtitle quality checks no longer enforce a hardcoded language constraint globally.
  - Language filtering is user-configurable via the `ACCEPTED_LANGS` environment variable (e.g. `ACCEPTED_LANGS=pt-BR` or `ACCEPTED_LANGS=pt-BR,en`). When unset or empty, the worker accepts all target language tags.
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
