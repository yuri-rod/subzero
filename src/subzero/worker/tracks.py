import json
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from .jellyfin import Media
from .srt import Cue, chunks, dump

BLOCK = 20
Progress = Callable[[str, int], None]

LANG_NAMES = {
    "pt-BR": "portugues do Brasil",
    "pt": "portugues",
    "en": "ingles",
    "es": "espanhol",
    "ja": "japones",
}


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
    wav = Path(tempfile.gettempdir()) / f"srt-{abs(hash(video_path))}.wav"
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
           "-f", "wav", str(wav)]
    run_ffmpeg(cmd, duration, progress, "extraindo audio", popen=popen)
    return str(wav)


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
            self.model = WhisperModel(self.name, device=self.device, compute_type=self.compute_type)
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
        total = max(1.0, float(getattr(info, "duration", 0) or 0))
        cues: list[Cue] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            cues.append(Cue(len(cues) + 1, float(seg.start), float(seg.end), text))
            progress("transcrevendo", int(min(99, float(seg.end) / total * 100)))
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
    # o gemma3:12b ocupa quase toda a VRAM da 3060; segurando ele carregado o whisper
    # do proximo job nao acha lugar. Meio minuto cobre os blocos de uma mesma legenda
    # e devolve a placa logo depois.
    KEEP_ALIVE = "30s"

    def __init__(self, url: str, model: str, http=None, keep_alive: str = KEEP_ALIVE):
        self.url = url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        if http is None:
            import httpx
            http = httpx.Client(timeout=600)
        self.http = http

    def release(self) -> None:
        """Pede ao ollama para largar o modelo agora. Best-effort: se falhar, o
        keep_alive derruba sozinho em seguida."""
        try:
            self.http.request("POST", f"{self.url}/api/generate",
                              json={"model": self.model, "prompt": "", "keep_alive": 0})
        except Exception:
            pass

    def translate_block(self, cues: list[Cue], target_lang: str) -> list[str]:
        target = LANG_NAMES.get(target_lang, target_lang)
        numbered = "\n".join(f"{i}. {c.text.replace(chr(10), ' ')}" for i, c in enumerate(cues, start=1))
        prompt = (
            f"Traduza para {target} ({target_lang}) as legendas numeradas abaixo.\n"
            f"Responda com exatamente {len(cues)} linhas, cada uma no formato 'numero. texto', "
            "sem comentarios, sem juntar linhas e sem pular numeros.\n\n" + numbered
        )
        last_err = None
        for attempt in range(3):
            try:
                r = self.http.request("POST", f"{self.url}/api/generate",
                                      json={"model": self.model, "prompt": prompt, "stream": False,
                                            "keep_alive": self.keep_alive,
                                            "options": {"temperature": 0.2}})
                if r.status_code == 200:
                    return self._lines(r.json().get("response", ""))
                last_err = f"ollama respondeu {r.status_code}: {r.text[:120]}"
            except Exception as e:
                last_err = f"erro ao chamar ollama: {e}"
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(last_err or "falha ao traduzir bloco com ollama")

    @staticmethod
    def _lines(response: str) -> list[str]:
        out = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)[.)]\s*(.+)$", line)
            if match:
                if int(match.group(1)) != len(out)+1:
                    return []
                out.append(match.group(2).strip())
        return out


def translate(cues: list[Cue], target_lang: str, ollama: Ollama, progress: Progress,
              strict: bool = False) -> list[Cue]:
    done: list[Cue] = []
    blocks = list(chunks(cues, BLOCK))
    fell_back = 0
    for n, block in enumerate(blocks, start=1):
        lines = ollama.translate_block(block, target_lang)
        # modelo perdeu ou inventou linha: mantem o original em vez de desalinhar a legenda
        if len(lines) != len(block):
            if strict:
                raise RuntimeError(f'o modelo nao preservou as linhas do bloco {n}')
            lines = [c.text for c in block]
            fell_back += 1
        for cue, text in zip(block, lines):
            done.append(Cue(cue.index, cue.start, cue.end, text or cue.text))
        progress(f"traduzindo bloco {n}/{len(blocks)}", int(n / len(blocks) * 100))
    # um bloco isolado desalinhar acontece; todos desalinharem e o ollama fora do ar
    if len(blocks) > 1 and fell_back == len(blocks):
        raise RuntimeError("o modelo nao traduziu nenhum bloco, descartando a saida")
    return done


def deliver(media: Media, cues: list[Cue], lang: str, jellyfin, bare: bool = False) -> str:
    if not cues:
        raise RuntimeError("legenda vazia, nada foi gravado")
    path = sidecar_path(media.path, lang, bare=bare)
    Path(path).write_text(dump(cues), encoding="utf-8")
    jellyfin.refresh(media.item_id)
    return path
