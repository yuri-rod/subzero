import json
import math
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from subzero.translate import MAX_RESPONSE_BYTES, _is_native_translation, _ollama_payload, _parse_lines, _parse_ollama_response, _translate_lines

from .jellyfin import Media
from .srt import Cue, chunks, dump

BLOCK = 20
Progress = Callable[[str, int], None]


def split_path(video_path: str) -> tuple[str, str, str]:
    sep = "\\" if "\\" in video_path else "/"
    folder, _, name = video_path.rpartition(sep)
    stem, _, ext = name.rpartition(".")
    return folder, (stem or name), sep


def sidecar_path(video_path: str, lang: str, bare: bool = False) -> str:
    """Caminho do sidecar. Com bare o arquivo fica com o nome exato do video, sem tag
    de idioma, que e como o Jellyfin trata a legenda como a padrao do item."""
    folder, stem, sep = split_path(video_path)
    name = f"{stem}.srt" if bare else f"{stem}.{lang}.srt"
    return f"{folder}{sep}{name}" if folder else name


def last_lines(text: str | None, count: int = 4) -> str:
    """O erro util do ffmpeg fica no fim; o comeco e sempre o banner de build.

    O corte tambem e pelo fim: uma unica linha gigante (dump de streams de um 2160p)
    comia os 300 caracteres inteiros e escondia justamente a linha do erro.
    """
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    return " | ".join(lines[-count:])[-300:]


def run_ffmpeg(cmd: list[str], duration: float, progress: Progress, phase: str,
               popen=subprocess.Popen) -> None:
    """Roda o ffmpeg lendo o -progress dele, porque num 4K de disco lento isso leva minutos.

    O stderr e drenado numa thread em vez de so ser lido depois do wait(). Um arquivo
    com muitas trilhas (2160p com 18 legendas, por exemplo) escreve banner e dump de
    streams alem do buffer do pipe; o ffmpeg trava escrevendo, o stdout para de vir e o
    laco abaixo espera para sempre por um processo que nunca mais anda.
    """
    proc = popen(cmd + ["-progress", "pipe:1", "-nostats"], stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, bufsize=1)
    errors: list[str] = []

    def drain() -> None:
        try:
            errors.append(proc.stderr.read() or "")
        except (OSError, ValueError):
            errors.append("")

    reader = threading.Thread(target=drain, name="ffmpeg-stderr", daemon=True)
    reader.start()

    total = max(1.0, duration)
    for line in proc.stdout:
        if line.startswith("out_time_ms="):
            done = int(line.split("=", 1)[1].strip() or 0) / 1_000_000
            progress(phase, int(min(99, done / total * 100)))
    code = proc.wait()
    reader.join(timeout=10)
    if code != 0:
        raise RuntimeError(f"ffmpeg falhou: {last_lines(errors[0] if errors else '')}")


def extract_embedded(media: Media, sub_index: int, lang: str, progress: Progress = lambda p, n: None,
                     popen=subprocess.Popen, bare: bool = False) -> str:
    out = sidecar_path(media.path, lang, bare=bare)
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", media.path, "-map", f"0:s:{sub_index}", "-vn", "-an",
           "-c:s", "srt", out]
    run_ffmpeg(cmd, media.duration, progress, "extraindo do arquivo", popen=popen)
    if Path(out).exists() and Path(out).stat().st_size == 0:
        Path(out).unlink()
        raise RuntimeError(f"a trilha {sub_index} nao rendeu nenhuma legenda")
    return out


# abaixo disso ninguem percebe, e mexer na legenda so adiciona ruido de arredondamento
MIN_OFFSET = 0.05


def audio_start_offset(video_path: str, run=subprocess.run) -> float:
    """Quanto o audio comeca depois do video no container.

    Alguns WEB-DL (Amazon e o caso classico) trazem o audio comecando alguns
    segundos depois do video. O ffmpeg zera o timestamp na hora de extrair o wav,
    entao o Whisper transcreve contra um relogio adiantado e a legenda inteira sai
    adiantada pelo mesmo tanto. Devolve o desvio para somar de volta nas cues.
    """
    cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time",
           "-of", "json", video_path]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 0.0
    if getattr(out, "returncode", 1) != 0:
        return 0.0
    try:
        streams = json.loads(out.stdout or "{}").get("streams", [])
    except ValueError:
        return 0.0

    def start_of(kind: str) -> float:
        for stream in streams:
            if stream.get("codec_type") == kind:
                try:
                    return float(stream.get("start_time"))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    offset = start_of("audio") - start_of("video")
    return offset if offset >= MIN_OFFSET else 0.0


def shift(cues: list[Cue], offset: float) -> list[Cue]:
    if not offset:
        return cues
    return [Cue(c.index, c.start + offset, c.end + offset, c.text) for c in cues]


def extract_audio(video_path: str, duration: float = 0, progress: Progress = lambda p, n: None,
                  popen=subprocess.Popen) -> str:
    with tempfile.NamedTemporaryFile(prefix="subzero-audio-", suffix=".wav", delete=False) as handle:
        wav = Path(handle.name)
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
           "-f", "wav", str(wav)]
    complete = False
    try:
        run_ffmpeg(cmd, duration, progress, "extraindo audio", popen=popen)
        complete = True
        return str(wav)
    finally:
        if not complete:
            wav.unlink(missing_ok=True)


class ModelHolder:
    """Segura o Whisper carregado e devolve a GPU antes de o gemma3 ser chamado."""

    def __init__(self, name: str, device: str = "cuda", compute_type: str | None = None):
        self.name = name
        self.device = device
        if compute_type is None:
            self.compute_type = "int8" if device == "cpu" else "float16"
        else:
            self.compute_type = compute_type
        self.model = None
        # a fila roda numa thread e o HTTP noutra: dois loads ao mesmo tempo estouram a VRAM
        self.lock = threading.Lock()

    def load(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            from faster_whisper.utils import download_model
            from huggingface_hub.errors import LocalEntryNotFoundError

            snapshot = Path(self.name).expanduser()
            if not snapshot.is_dir():
                try:
                    snapshot = Path(download_model(self.name, local_files_only=True))
                except LocalEntryNotFoundError as err:
                    raise RuntimeError(f"Whisper model {self.name!r} is not cached locally; prepare the model before starting the worker") from err
            for filename in ("model.bin", "config.json", "tokenizer.json"):
                try:
                    with (snapshot / filename).open("rb") as handle:
                        handle.read(1)
                except FileNotFoundError as err:
                    raise RuntimeError(f"Whisper cache is incomplete: missing {filename}; prepare the model before starting the worker") from err
                except PermissionError as err:
                    raise RuntimeError(f"Whisper cache is not readable: {snapshot / filename}") from err
            self.model = WhisperModel(str(snapshot), device=self.device, compute_type=self.compute_type,
                                      local_files_only=True)
        return self.model

    def unload(self) -> None:
        if self.model is not None:
            try:
                if hasattr(self.model, "model") and hasattr(self.model.model, "unload_model"):
                    self.model.model.unload_model()
            except Exception:
                pass
        self.model = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def transcribe(audio_path: str, holder: ModelHolder, progress: Progress) -> tuple[list[Cue], str]:
    with holder.lock:
        return _transcribe(audio_path, holder, progress)


def _transcribe(audio_path: str, holder: ModelHolder, progress: Progress) -> tuple[list[Cue], str]:
    model = holder.load()
    try:
        segments, info = model.transcribe(audio_path, vad_filter=True, beam_size=5)
        duration = float(getattr(info, "duration", 0) or 0)
        bounded = math.isfinite(duration) and duration > 0
        total = max(1.0, duration) if bounded else 1.0
        cues: list[Cue] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            start, end = float(seg.start), float(seg.end)
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            start = max(0.0, start)
            if bounded:
                end = min(duration, end)
            if end <= start:
                continue
            cues.append(Cue(len(cues) + 1, start, end, text))
            progress("transcrevendo", int(min(99, end / total * 100)))
        language = getattr(info, "language", "") or "und"
    finally:
        holder.unload()

    if not cues:
        raise RuntimeError("o audio nao rendeu nenhuma fala, nada a gravar")
    if _stuck_in_a_loop(cues):
        raise RuntimeError("o whisper travou repetindo a mesma fala, descartando a saida")
    return cues, language


def _stuck_in_a_loop(cues: list[Cue]) -> bool:
    """Falha classica do faster-whisper em audio ruim: repete a mesma frase centenas de vezes."""
    if len(cues) < 12:
        return False
    texts = [c.text.strip().lower() for c in cues]
    _, count = Counter(texts).most_common(1)[0]
    return count / len(texts) > 0.5


class Ollama:
    KEEP_ALIVE = "2m"

    def __init__(self, url: str, model: str, http=None, keep_alive: str = KEEP_ALIVE,
                 num_ctx: int = 4096, num_predict: int = 2048):
        self.url = url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        if num_ctx < 1 or num_predict < 1:
            raise ValueError("Ollama context and output limits must be positive")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        if http is None:
            import httpx
            http = httpx.Client(timeout=600)
        self.http = http

    def ensure_available(self) -> None:
        import httpx
        try:
            response = self.http.request("POST", f"{self.url}/api/show",
                                         json={"model": self.model}, timeout=5)
        except (httpx.HTTPError, OSError):
            raise RuntimeError("Ollama is unavailable; start the local service before translating") from None
        if response.status_code == 404:
            raise RuntimeError("Ollama model is not installed; install the configured model before translating")
        if response.status_code != 200:
            raise RuntimeError(f"Ollama model check failed ({response.status_code})")

    def release(self) -> None:
        """Pede ao ollama para largar o modelo agora. Best-effort: se falhar, o
        keep_alive derruba sozinho em seguida."""
        import httpx
        try:
            self.http.request("POST", f"{self.url}/api/generate",
                              json={"model": self.model, "prompt": "", "keep_alive": 0}, timeout=5)
        except (httpx.HTTPError, OSError):
            pass

    def translate_block(self, cues: list[Cue], target_lang: str, source_lang: str | None = None, *, context=None) -> list[str]:
        import httpx
        if _is_native_translation(self.model) and len(cues) > 1:
            return [line for index, cue in enumerate(cues)
                    for line in _translate_lines([cue], target_lang, self, source_lang=source_lang, context=context or {
                        "previous": "\n".join(c.text for c in cues[max(0, index - 2):index]),
                        "following": "\n".join(c.text for c in cues[index + 1:index + 3]),
                    })]
        payload = _ollama_payload(cues, target_lang, self.model, self.keep_alive,
                                  self.num_ctx, self.num_predict, source_lang, context=context)
        last_err = None
        for attempt in range(3):
            try:
                r = self.http.request("POST", f"{self.url}/api/generate",
                                      json=payload)
                if r.status_code == 200:
                    if len(getattr(r, "content", b"")) > MAX_RESPONSE_BYTES:
                        raise RuntimeError("Ollama response exceeds the subtitle size limit")
                    try:
                        body = r.json()
                    except (ValueError, UnicodeError):
                        return []
                    return _parse_ollama_response(body, native=_is_native_translation(self.model))
                if r.status_code == 404:
                    raise RuntimeError("Ollama model is not installed")
                if 400 <= r.status_code < 500:
                    raise RuntimeError(f"Ollama rejected the translation request ({r.status_code})")
                last_err = f"Ollama returned status {r.status_code}"
            except (httpx.HTTPError, OSError):
                last_err = "Ollama is unavailable or the translation request timed out"
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(last_err or "falha ao traduzir bloco com ollama")

    @staticmethod
    def _lines(response: str) -> list[str]:
        return _parse_lines(response)


def translate(cues: list[Cue], target_lang: str, ollama: Ollama, progress: Progress,
              strict: bool = False, source_lang: str | None = None, *, context=None) -> list[Cue]:
    done: list[Cue] = []
    blocks = list(chunks(cues, BLOCK))
    for n, block in enumerate(blocks, start=1):
        try:
            options = {'context': context} if context is not None else {}
            lines = _translate_lines(block, target_lang, ollama, source_lang=source_lang, **options)
        except RuntimeError as err:
            raise RuntimeError(f'Falha no bloco {n}: {err}') from None
        for cue, text in zip(block, lines):
            done.append(Cue(cue.index, cue.start, cue.end, text))
        progress(f"traduzindo bloco {n}/{len(blocks)}", int(n / len(blocks) * 100))
    return done


def deliver(media: Media, cues: list[Cue], lang: str, jellyfin, bare: bool = False) -> str:
    if not cues:
        raise RuntimeError("legenda vazia, nada foi gravado")
    from .service import same_language
    path = sidecar_path(media.path, lang, bare=bare)
    target = Path(path)
    target.write_text(dump(cues), encoding="utf-8")
    video = Path(media.path)
    embedded_langs = {getattr(s, "lang", "").lower() for s in getattr(media, "embedded", [])}
    for p in video.parent.glob(f"{video.stem}*.srt"):
        try:
            if p.resolve() == target.resolve():
                continue
            if not bare and p.name == f"{video.stem}.srt":
                p.unlink(missing_ok=True)
            tag = p.name[len(video.stem) + 1:-4].lower()
            if tag and any(same_language(tag, el) for el in embedded_langs if el):
                p.unlink(missing_ok=True)
        except OSError:
            pass
    jellyfin.refresh(media.item_id)
    return path
