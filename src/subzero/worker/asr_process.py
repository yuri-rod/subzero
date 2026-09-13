"""Run transcription in a child whose exit releases native model resources."""

import json
import math
import queue
import subprocess
import sys
import tempfile
import threading
import time

from subzero.compute import ComputeShutdownError, compute_lease_fd
from .srt import Cue


def transcribe_in_process(audio_path, holder, progress):
    if getattr(holder, "model", None) is not None:
        raise RuntimeError("Whisper is already loaded in the worker; restart before isolated transcription")
    argv = [sys.executable, "-m", "subzero.worker.asr_process", str(audio_path),
            holder.name, holder.device, holder.compute_type]
    messages = queue.Queue(maxsize=64)
    stopped = threading.Event()
    cues = []
    language = None
    percent = 0
    received = 0
    last_progress = time.monotonic()
    lease = compute_lease_fd()
    with tempfile.TemporaryFile(mode="w+b") as errors:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors,
                                pass_fds=() if lease is None else (lease,))

        def send(message):
            while not stopped.is_set():
                try:
                    messages.put(message, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def read_messages():
            try:
                while not stopped.is_set():
                    line = proc.stdout.readline(65_537)
                    send(line)
                    if not line:
                        return
            except (OSError, ValueError) as err:
                send(err)

        reader = threading.Thread(target=read_messages, name="whisper-output", daemon=True)
        try:
            reader.start()
            while True:
                try:
                    line = messages.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - last_progress >= 5:
                        progress("transcrevendo", percent)
                        last_progress = time.monotonic()
                    continue
                if isinstance(line, Exception):
                    raise RuntimeError("Whisper child output stream failed") from line
                if not line:
                    break
                received += len(line)
                if len(line) > 65_536 or received > 16_000_000:
                    raise RuntimeError("Whisper child output exceeds the subtitle size limit")
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError) as err:
                    raise RuntimeError("Whisper child returned invalid JSON") from err
                if not isinstance(event, dict) or language is not None:
                    raise RuntimeError("Whisper child returned an invalid event")
                kind = event.get("event")
                if kind == "progress":
                    percent = event.get("percent")
                    if type(percent) is not int or not 0 <= percent <= 99:
                        raise RuntimeError("Whisper child returned invalid progress")
                    progress("transcrevendo", percent)
                    last_progress = time.monotonic()
                elif kind == "cue":
                    start, end, text = event.get("start"), event.get("end"), event.get("text")
                    if (type(start) not in (int, float) or type(end) not in (int, float)
                            or not math.isfinite(start + end) or start < 0 or end <= start
                            or not isinstance(text, str) or not text.strip()
                            or type(event.get("index")) is not int or event["index"] != len(cues) + 1
                            or (cues and start < cues[-1].start)):
                        raise RuntimeError("Whisper child returned an invalid subtitle cue")
                    cues.append(Cue(len(cues) + 1, start, end, text))
                elif kind == "complete":
                    language = event.get("language")
                    if (not isinstance(language, str) or not language or len(language) > 32
                            or not cues or type(event.get("cues")) is not int or event["cues"] != len(cues)):
                        raise RuntimeError("Whisper child returned an incomplete transcription")
                elif kind == "error" and isinstance(event.get("message"), str):
                    raise RuntimeError(f"Whisper transcription failed: {event['message'][:500]}")
                else:
                    raise RuntimeError("Whisper child returned an unknown event")
            while True:
                try:
                    code = proc.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - last_progress >= 5:
                        progress("transcrevendo", percent)
                        last_progress = time.monotonic()
            if code != 0:
                errors.seek(max(0, errors.tell() - 1000))
                reason = errors.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Whisper child exited with status {code}: {reason}")
            if language is None:
                raise RuntimeError("Whisper child exited without a complete transcription")
            return cues, language
        finally:
            stopped.set()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired as err:
                        raise ComputeShutdownError("Whisper child did not exit after termination",
                                                   pids=(proc.pid,)) from err
            if reader.ident is not None:
                reader.join(timeout=2)
            proc.stdout.close()
            if reader.is_alive():
                raise RuntimeError("Whisper output reader did not stop after child exit")


def main():
    from .tracks import ModelHolder, _transcribe
    if len(sys.argv) != 5:
        raise RuntimeError("Whisper child requires audio, model, device, and compute type")
    audio, name, device, compute_type = sys.argv[1:]

    def send(event):
        print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        cues, language = _transcribe(audio, ModelHolder(name, device, compute_type),
                                    lambda phase, percent: send({"event": "progress", "percent": percent}))
        for cue in cues:
            send({"event": "cue", "index": cue.index, "start": cue.start, "end": cue.end, "text": cue.text})
        send({"event": "complete", "language": language, "cues": len(cues)})
    except (ImportError, OSError, RuntimeError, ValueError) as err:
        send({"event": "error", "message": str(err)[:500]})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
