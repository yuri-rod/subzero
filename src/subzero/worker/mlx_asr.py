"""Apple Silicon transcription through mlx-whisper.

Same turbo weights as the CPU path, roughly 3x faster on 90s of audio (11s vs
35s measured on the S47 sample) with equal-or-better text. MlxModel speaks the
faster-whisper dialect the rest of the worker already expects, so _transcribe
needs no changes. Selected with WHISPER_DEVICE=mlx.
"""

from pathlib import Path
from types import SimpleNamespace


def mlx_repo(name):
    """Short Whisper names map to mlx-community; full repos and local snapshot
    paths pass through untouched."""
    if "/" in name or Path(name).expanduser().is_dir():
        return name
    return f"mlx-community/whisper-{name}"


def resolve_snapshot(repo):
    """Local snapshot for an mlx repo. Same contract as the faster-whisper
    path: the worker never downloads mid-job, it fails fast instead."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot = Path(snapshot_download(repo, local_files_only=True))
    except LocalEntryNotFoundError as err:
        raise RuntimeError(f"Whisper model {repo!r} is not cached locally; prepare the model before starting the worker") from err
    # whisper-large-v3-turbo ships exactly these two plus a README; decoding
    # uses the tokenizer bundled in mlx_whisper itself, no tokenizer.json.
    for filename in ("config.json", "weights.safetensors"):
        try:
            with (snapshot / filename).open("rb") as handle:
                handle.read(1)
        except FileNotFoundError as err:
            raise RuntimeError(f"Whisper cache is incomplete: missing {filename}; prepare the model before starting the worker") from err
        except PermissionError as err:
            raise RuntimeError(f"Whisper cache is not readable: {snapshot / filename}") from err
    return snapshot


class MlxModel:
    def __init__(self, snapshot):
        self.snapshot = str(snapshot)

    def transcribe(self, audio_path, **kwargs):
        import contextlib
        import sys

        import mlx_whisper

        # faster-whisper options with no mlx equivalent; the turbo defaults
        # (temperature fallback, no-speech filtering) already match the
        # quality the worker expects from the CPU path.
        kwargs.pop("vad_filter", None)
        kwargs.pop("beam_size", None)
        # mlx_whisper prints "Detected language: ..." to stdout, and the
        # isolated child speaks JSON-lines on stdout; keep the chatter on
        # stderr where the parent only reads it on failure.
        # The temperature fallback samples unseeded: same audio, different
        # segments every run. Seed it so transcriptions are reproducible.
        # (ImportError only happens with a stubbed mlx_whisper in tests;
        # the real package hard-depends on mlx.)
        try:
            import mlx.core as mx
        except ImportError:
            pass
        else:
            mx.random.seed(0)
        with contextlib.redirect_stdout(sys.stderr):
            out = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=self.snapshot,
                                         verbose=False, **kwargs)
        segs = [SimpleNamespace(start=float(seg["start"]), end=float(seg["end"]),
                                text=seg["text"])
                for seg in out.get("segments", [])]
        duration = max((seg.end for seg in segs), default=0.0)
        info = SimpleNamespace(language=out.get("language") or "und", duration=duration)
        return iter(segs), info
