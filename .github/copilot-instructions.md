# Repository guidance

## Build and test

- Install the development/test dependencies with `python -m pip install -e ".[dev]"`.
- Run the full test suite with `pytest -v` (the same command used by CI).
- Run one test with `pytest tests/test_cli.py::test_python_m_subzero_supports_version`, or a test module with `pytest tests/test_cli.py`.
- CI tests Python 3.10–3.13 on Ubuntu, macOS, and Windows.
- No separate build or lint command is configured in the project.

## Architecture

- This is a `src/`-layout Python package. `subzero.__init__` exposes the public library API; `subzero.cli` composes the command-line interface from feature modules such as cleanup (`core`), conversion, extraction, timing, translation, and OCR.
- The base package has no runtime dependencies. Speech-sync features use the `sync` extra; the FastAPI subtitle worker uses the `worker` extra; `dev` provides test dependencies and optional feature dependencies.
- `subzero.worker` is the optional Jellyfin/YUCAST daemon: `api` exposes HTTP endpoints, `jobs` persists and schedules work in SQLite, and `service`/`syncflow` coordinate subtitle operations through integration clients. `src/srtworker` is a compatibility package that re-exports the worker implementation.
- The README documents user-visible behavior and architecture constraints. Consult it before changing CLI options, configuration, timing verification, or worker workflows.

## Repository-specific conventions

- Subtitle cleanup preserves intact human line breaks by default. It repairs collapsed dialogue and enforces the configured line-length limit (42 characters by default); only reflow all cues when explicitly requested.
- Keep optional dependencies optional: feature-specific integrations belong behind their corresponding extras rather than becoming base-package requirements.
- Local Ollama calls must run inside `compute_phase("ollama", ollama_url=...)`. This coordinates managed daemon lifecycle and shared compute-lock ownership across CLI, worker, OCR, and transcription paths.
- Preserve safe output behavior in file-changing flows: retain dry-run and backup options, and use the existing temporary-file/atomic-replacement patterns when writing generated files.
- Timing verification is evidence-based and read-only. Automatic correction is conservative and must be re-verified before accepting the corrected subtitle; container start-time skew alone is not proof of dialogue synchronization.
