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

## Provider limitations and operating rules

DeepL Free uses a fixed Free endpoint and checks the episode budget before translation. With `TRANSLATION_FALLBACK` configured, an episode that exceeds the allowance uses that provider; otherwise the queue pauses. Account usage can change outside this worker, so quota exhaustion during translation still needs a separate check. Paid usage remains disabled.

LibreTranslate and the optional providers can mistranslate ambiguous terms, idioms, names or compressed dialogue, and omit details. Target-language and formatting checks do not detect every semantic error. Keep representative source/output review separate from provider availability and speed measurements, including when DeepL is selected.

Before removing a selected Ollama model or runtime, update provider settings and verify the worker path. Translation settings form part of cache identity, so changed settings start separate caches.

Keep discarded benchmark environments and models separate from active runtimes, proof files and resumable caches. Consolidate duplicate library copies only after verifying media health and retaining the chosen files and sidecars. Retain job IDs and backup provenance when retrying; never infer success from queue exhaustion or blindly submit the same work again.
