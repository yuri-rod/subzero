# TODO

Keep code checks, deployed behavior, installed-file verification, and translation quality as separate results. Passing tests or an audio timing check does not close full-output validation.

## Open work

- [ ] Validate the wrapped SDH cleanup through a complete worker retry. A multiline sound description could survive cleanup as an extra sentence anchor and make a valid short translation fail redistribution. Focused source-cleanup tests pass; complete installation still needs verification.
- [ ] Validate the corrected native full-frame and region-of-interest row reconciliation on complete source scans when one physical caption is split into fragments. A missing censorship bar can otherwise produce a duplicated suffix. Preserve observed bar position and real dialogue without guessing hidden words.
- [ ] Validate consistent coarse/dense region admission through a complete retry. A bounded replay traced an exact-timestamp conflict to selective coarse ROI checks despite identical input frames. The fix uses the same admission path and retains rejection for real conflicts.
- [ ] Verify complete repair and rebuild outputs after these fixes, including original source anchors, recovered caption timing, target language, backups, installed hashes, and media-server delivery.
- [ ] Audit the generic CLI translation defaults separately from worker provider selection. Some OCR paths still default to `subzero/hy-mt2:7b` when no translation client is supplied. Removing a local model does not remove those defaults; do not silently substitute a provider.
- [ ] Keep external queue monitors compatible with exact installation digests, fresh health observations, approved deployed versions, and explicit holds. Monitor corrections need a process restart and a fresh observation, not just a file edit.

## Verified protections to retain

- [x] Provider output passes structure and target-language checks before installation. Translation does not silently switch providers on failure.
- [x] Repair requires a verified English subtitle source. Embedded language metadata alone is insufficient. Rebuild supplies isolated transcription when no verified text source exists, while retaining the independent audio timing reference.
- [x] Full-video caption recovery can preserve distinct simultaneous captions and compose translated display intervals without overlapping output cues.
- [x] Native caption admission filters screen labels, credits and title layouts. Dense timing, protected names/numbers/negation, and positioned censorship-marker checks stop unresolved loss or flicker for review.
- [x] A shared compute policy serializes participating OCR, Ollama and Whisper operations. Child exit and daemon shutdown must complete before ownership changes; CPU transcription also follows this lifecycle.
- [x] Candidate staging, source-change checks, backups and atomic installation protect the target before replacement. A later media-server refresh error can still occur after replacement, so terminal job state alone is insufficient.
- [x] Literal sidecar matching handles bracketed release names without expanding them as glob patterns.
- [x] Explicit environment-file selection fails when the file is absent or the option has no argument. Obsolete automatic configuration discovery was removed without breaking explicit compatibility paths.
- [x] The native LibreTranslate runtime is isolated, hash-pinned and verified before use. Translation runs on CPU without network access. Installer support remains limited to the documented interpreter and platform.

These checks cover their documented scope. They do not prove that every caption, translation or complete media file is correct.

## Provider limitations and operating rules

The selected local LibreTranslate engine can still mistranslate ambiguous terms, idioms, names or compressed dialogue, and omit details. Target-language and formatting checks do not detect every semantic error. Keep representative source/output review separate from provider availability and speed measurements.

Optional Ollama providers remain supported. Compatibility code and examples are not evidence that a model is installed or selected. Change provider/model settings deliberately, invalidate affected caches, and test the actual worker path before removing a selected runtime.

Keep discarded benchmark environments and models separate from active runtimes, proof files and resumable caches. Consolidate duplicate library copies only after verifying media health and retaining the chosen files and sidecars. Retain job IDs and backup provenance when retrying; never infer success from queue exhaustion or blindly submit the same work again.
