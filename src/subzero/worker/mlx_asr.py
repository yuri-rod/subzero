"""Apple Silicon transcription through mlx-whisper.

Same turbo weights as the CPU path, roughly 2.5x faster on 90s of episode
audio (8.4s vs 21.6s on an M3 Pro, large-v3-turbo, 92% word overlap). MlxModel
speaks the faster-whisper dialect the rest of the worker already expects, and
it gates decoding with the same Silero VAD spans faster-whisper uses: without
that gate the model invents dialogue over music and montage, which the CPU path
never emits. Selected with WHISPER_DEVICE=mlx.
"""

from pathlib import Path
from types import SimpleNamespace

SAMPLE_RATE = 16000
# faster-whisper's vad_filter defaults: chunk_length 30, min_silence 160 ms
VAD_MAX_CLIP_SECONDS = 30.0
VAD_MIN_SILENCE_MS = 160


def speech_chunks(audio_path):
    """Speech audio in <=30s chunks plus each chunk's original-time mapping.

    Mirrors faster-whisper's vad_filter path: collect_chunks concatenates the
    Silero spans, and segment timestamps are restored to the original timeline
    afterwards. mlx-whisper's own clip_timestamps path emits timestamps past the
    audio and repetition loops, so the worker does the mapping itself.
    """
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    samples = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    options = VadOptions(max_speech_duration_s=VAD_MAX_CLIP_SECONDS,
                         min_silence_duration_ms=VAD_MIN_SILENCE_MS)
    chunks = []
    parts = []
    pieces = []
    for span in get_speech_timestamps(samples, options):
        if parts and (parts[-1][1] + (span["end"] - span["start"]) / SAMPLE_RATE
                      > VAD_MAX_CLIP_SECONDS):
            chunks.append({"audio": np.concatenate(pieces), "parts": parts})
            parts, pieces = [], []
        local = parts[-1][1] if parts else 0.0
        pieces.append(samples[span["start"]:span["end"]])
        parts.append((local, local + (span["end"] - span["start"]) / SAMPLE_RATE,
                      span["start"] / SAMPLE_RATE, span["end"] / SAMPLE_RATE))
    if parts:
        chunks.append({"audio": np.concatenate(pieces), "parts": parts})
    return chunks


def restore_time(value, parts):
    """Map a chunk-local timestamp back to the original timeline."""
    if value <= parts[0][0]:
        return parts[0][2]
    for local_start, local_end, original_start, original_end in parts:
        if value <= local_end:
            return original_start + (value - local_start)
    return parts[-1][3]


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

        # beam_size has no mlx equivalent.
        vad_filter = kwargs.pop("vad_filter", None)
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
        if vad_filter:
            chunks = speech_chunks(audio_path)
            if not chunks:
                raise RuntimeError("VAD found no speech in this audio")
        else:
            chunks = [{"audio": str(audio_path), "parts": None}]
        segs = []
        language = ""
        with contextlib.redirect_stdout(sys.stderr):
            for chunk in chunks:
                out = mlx_whisper.transcribe(chunk["audio"], path_or_hf_repo=self.snapshot,
                                             verbose=False, **kwargs)
                language = language or out.get("language") or ""
                parts = chunk["parts"]
                for seg in out.get("segments", []):
                    start, end = float(seg["start"]), float(seg["end"])
                    if parts is not None:
                        start, end = restore_time(start, parts), restore_time(end, parts)
                    segs.append(SimpleNamespace(start=start, end=end, text=seg["text"]))
        duration = max((seg.end for seg in segs), default=0.0)
        info = SimpleNamespace(language=language or "und", duration=duration)
        return iter(segs), info
