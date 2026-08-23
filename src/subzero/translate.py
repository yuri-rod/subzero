"""Local LLM subtitle translation for Subzero."""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from .core import read, CUE, TIMECODE
from .convert import Cue, parse_srt, dump_srt


LANG_NAMES = {
    "pt-BR": "portugues do Brasil",
    "pt": "portugues",
    "en": "English",
    "es": "espanhol",
    "fr": "francais",
    "de": "Deutsch",
    "it": "italiano",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}


def chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class OllamaClient:
    def __init__(self, url: str = "http://127.0.0.1:11434", model: str = "gemma3:12b", timeout: int = 120):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def translate_block(self, cues: list[Cue], target_lang: str) -> list[str]:
        target_name = LANG_NAMES.get(target_lang, target_lang)
        numbered = "\n".join(f"{i}. {c.text.replace(chr(10), ' ')}" for i, c in enumerate(cues, start=1))
        prompt = (
            f"Translate the following numbered subtitle lines into {target_name} ({target_lang}).\n"
            f"Respond with exactly {len(cues)} lines, each in the format 'number. text', "
            "without comments, without combining lines, and without skipping numbers.\n\n" + numbered
        )
        req_data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8"))
                        return self._parse_lines(body.get("response", ""))
                    last_err = f"Ollama returned status {resp.status}"
            except (urllib.error.URLError, TimeoutError) as err:
                last_err = f"Ollama connection error: {err}"
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(last_err or "Failed to translate block with Ollama")

    @staticmethod
    def _parse_lines(response_text: str) -> list[str]:
        out = []
        for line in response_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)[.)]\s*(.+)$", line)
            if m:
                out.append(m.group(2).strip())
        return out


def translate_cues(
    cues: list[Cue],
    target_lang: str,
    client: OllamaClient,
    batch_size: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> list[Cue]:
    blocks = list(chunks(cues, batch_size))
    translated: list[Cue] = []
    for idx, block in enumerate(blocks, start=1):
        lines = client.translate_block(block, target_lang)
        if len(lines) != len(block):
            lines = [c.text for c in block]
        for cue, text in zip(block, lines):
            translated.append(Cue(cue.start, cue.end, text or cue.text))
        if progress:
            progress(idx, len(blocks))
    return translated


def translate_file(
    path: str | Path,
    target_lang: str = "pt-BR",
    output: str | Path | None = None,
    model: str = "gemma3:12b",
    url: str = "http://127.0.0.1:11434",
    batch_size: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    p = Path(path)
    content, _ = read(p)
    cues = parse_srt(content)
    if not cues:
        raise ValueError(f"No cues found in {p}")

    client = OllamaClient(url=url, model=model)
    translated = translate_cues(cues, target_lang, client, batch_size=batch_size, progress=progress)

    if output:
        out_path = Path(output)
    else:
        stem = p.stem
        if stem.endswith(".en") or stem.endswith(".eng"):
            base_stem = stem.rsplit(".", 1)[0]
            out_path = p.parent / f"{base_stem}.{target_lang}.srt"
        else:
            out_path = p.parent / f"{stem}.{target_lang}.srt"

    out_path.write_text(dump_srt(translated), encoding="utf-8")
    return out_path
