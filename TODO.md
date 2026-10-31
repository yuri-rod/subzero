# TODO

Track full media runs separately from code tests. For each run, record the installed subtitle, backup, media-server response, and a review of the translated dialogue.

## Open work

- [ ] Validate the wrapped SDH cleanup through a complete worker retry. A multiline sound description could survive cleanup as an extra sentence anchor and make a valid short translation fail redistribution. Focused source-cleanup tests pass; complete installation still needs verification.
- [ ] Validate the corrected native full-frame and region-of-interest row reconciliation on complete source scans when one physical caption is split into fragments. A missing censorship bar can otherwise produce a duplicated suffix. Preserve observed bar position and real dialogue without guessing hidden words.
- [ ] Validate consistent coarse/dense region admission through a complete retry. A bounded replay traced an exact-timestamp conflict to selective coarse ROI checks despite identical input frames. The fix uses the same admission path and retains rejection for real conflicts.
- [ ] Verify complete repair and rebuild outputs after these fixes, including original source anchors, recovered caption timing, target language, backups, installed hashes, and media-server delivery.
- [ ] Audit the generic CLI translation defaults separately from worker provider selection. Some OCR paths still default to `subzero/hy-mt2:7b` when no translation client is supplied. Removing a local model does not remove those defaults; do not silently substitute a provider.
- [ ] Keep external queue monitors compatible with exact installation digests, fresh health observations, approved deployed versions, and explicit holds. Monitor corrections need a process restart and a fresh observation, not just a file edit.
- [ ] Verify a complete DeepL Free episode through translation, installation, and served-subtitle checks. Exercise quota holds separately from output failures and retain resumable caches when the remaining allowance cannot fit an episode.

## Verified protections to retain

- [x] Verified translation workflows check output structure and target language before installation. Provider changes require an explicit `TRANSLATION_FALLBACK` setting.
- [x] Repair requires a verified English subtitle source. Embedded language metadata alone is insufficient. Rebuild supplies isolated transcription when no verified text source exists, while retaining the independent audio timing reference.
- [x] Full-video caption recovery can preserve distinct simultaneous captions and compose translated display intervals without overlapping output cues.
- [x] Native caption admission filters screen labels, credits and title layouts. Dense timing, protected names/numbers/negation, and positioned censorship-marker checks stop unresolved loss or flicker for review.
- [x] A shared compute policy serializes participating OCR, Ollama and Whisper operations. Child exit and daemon shutdown must complete before ownership changes; CPU transcription also follows this lifecycle.
- [x] Candidate staging, source-change checks, backups and atomic installation protect the target before replacement. A later media-server refresh error can still occur after replacement, so terminal job state alone is insufficient.
- [x] Literal sidecar matching handles bracketed release names without expanding them as glob patterns.
- [x] Explicit environment-file selection fails when the file is absent or the option has no argument. Obsolete automatic configuration discovery was removed without breaking explicit compatibility paths.
- [x] The native LibreTranslate runtime is isolated, hash-pinned and verified before use. Translation runs on CPU without network access. Installer support remains limited to the documented interpreter and platform.
- [x] Multi-track embedded subtitle selection filters commentary, forced, and signs or songs tracks in extraction and worker flows, ranking candidate streams by stream metadata, cue density, and speech-timing overlap against audio references.
- [x] Fine-grained subprocess cancellation reaps FFmpeg, Whisper, and Vision OCR process groups immediately upon job cancellation or timeout, ensuring orphaned processes never retain the shared compute lease flock.
- [x] Piecewise synchronization detects segmented constant offsets across commercial breaks and act boundaries with per-segment speech-anchor confidence thresholds, ensuring monotonic cue ordering across segment transitions.

## Prioritized engineering roadmap

Estimated effort below reflects implementation, automated regression coverage, and end-to-end media verification (remaining: 18 to 23 hours).

1. [x] **Multi-track embedded subtitle selection and commentary filtering (completed)**
   - Inspect all embedded text subtitle streams in [`SyncFlow`](src/subzero/worker/syncflow.py) and [`extract.py`](src/subzero/extract.py) instead of stopping at the first matching language tag when multiple tracks exist (full dialogue, SDH, commentary, signs/songs).
   - Rank candidate streams by cue density, speech-timing overlap against [`reference.py`](src/subzero/reference.py), and stream title metadata so commentary or sparse forced tracks are not selected ahead of main dialogue.

2. [x] **Fine-grained subprocess cancellation for FFmpeg, Whisper, and Vision OCR (completed)**
   - Propagate job cancellation tokens into [`tracks.run_ffmpeg`](src/subzero/worker/tracks.py), [`asr_process.py`](src/subzero/worker/asr_process.py), and [`ocr.py`](src/subzero/ocr.py) child processes so `DELETE /jobs/{id}` terminates active media extraction or transcription promptly instead of waiting for the current phase to finish.
   - Ensure child process groups are reaped cleanly before releasing the shared compute lease.

3. [x] **Piecewise drift and commercial-break resynchronization (completed)**
   - Extend [`timing.py`](src/subzero/timing.py) and [`sync.py`](src/subzero/sync.py) beyond global linear offset and standard framerate ratios to detect segmented constant offsets across act breaks or recut broadcast masters.
   - Require per-segment speech-anchor confidence thresholds before applying piecewise shifts so sparse music or silence intervals do not skew dialogue timing.

4. **End-to-end worker verification for wrapped SDH cleanup and OCR ROI reconciliation (5 to 6 hours)**
   - Validate multiline SDH stripping, coarse/dense Vision OCR region admission, and censorship-bar fragment joining across complete episode retries in [`SyncFlow`](src/subzero/worker/syncflow.py).
   - Record deterministic fixture replays covering sentence-unit redistribution, backup retention, and Jellyfin library refresh failure recovery.

5. **CLI and worker translation provider configuration unification (4 to 5 hours)**
   - Align standalone CLI translation defaults in [`cli.py`](src/subzero/cli.py), [`translate.py`](src/subzero/translate.py), and [`ocr.py`](src/subzero/ocr.py) with [`worker/config.py`](src/subzero/worker/config.py) so `subzero translate` and `subzero fill-gaps` can read the same provider, model, and fallback settings without hardcoded model names.
   - Add dry-run provider preflight checks that verify model presence and endpoint reachability before running audio extraction or OCR.

6. **DeepL Free mid-episode quota exhaustion and resume recovery (4 to 5 hours)**
   - Handle mid-translation HTTP 456 quota errors in [`deepl.py`](src/subzero/worker/deepl.py) and [`SyncFlow`](src/subzero/worker/syncflow.py) by flushing completed blocks to [`translate_cache.py`](src/subzero/worker/translate_cache.py) and transitioning the job to `paused` without losing partial progress.
   - Add automated tests for partial-episode cache resume after quota reset or explicit fallback switch.

7. **Cross-platform OCR fallback for Linux and Windows workers (5 to 7 hours)**
   - Add an optional cross-platform burned-in caption backend behind the existing [`ocr.py`](src/subzero/ocr.py) frame-sampling and timeline-coalescing pipeline for non-macOS hosts where Apple Vision (`swiftc`) is unavailable.
   - Keep the Swift Vision bridge as the default on macOS and gate the cross-platform backend behind an optional dependency extra.

## Provider limitations and operating rules

DeepL Free uses a fixed Free endpoint and checks the episode budget before translation. With `TRANSLATION_FALLBACK` configured, an episode that exceeds the allowance uses that provider; otherwise the queue pauses. Account usage can change outside this worker, so quota exhaustion during translation still needs a separate check. Paid usage remains disabled.

LibreTranslate and the optional providers can mistranslate ambiguous terms, idioms, names or compressed dialogue, and omit details. Target-language and formatting checks do not detect every semantic error. Keep representative source/output review separate from provider availability and speed measurements, including when DeepL is selected.

Before removing a selected Ollama model or runtime, update provider settings and verify the worker path. Translation settings form part of cache identity, so changed settings start separate caches.

Keep discarded benchmark environments and models separate from active runtimes, proof files and resumable caches. Consolidate duplicate library copies only after verifying media health and retaining the chosen files and sidecars. Retain job IDs and backup provenance when retrying; never infer success from queue exhaustion or blindly submit the same work again.
