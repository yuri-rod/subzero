"""Subtitle translation engine supporting local Ollama LLMs and OpenAI-compatible APIs."""

from __future__ import annotations

import json
import os
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
    "ru": "Russian",
}


def chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


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


class OllamaClient:
    """Client for local Ollama HTTP API."""

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
                        return _parse_lines(body.get("response", ""))
                    last_err = f"Ollama returned status {resp.status}"
            except (urllib.error.URLError, TimeoutError) as err:
                last_err = f"Ollama connection error: {err}"
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(last_err or "Failed to translate block with Ollama")


class OpenAIClient:
    """Client for OpenAI-compatible chat completions APIs (OpenAI, Groq, OpenRouter, DeepSeek, LocalAI, vLLM)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 120,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def translate_block(self, cues: list[Cue], target_lang: str) -> list[str]:
        target_name = LANG_NAMES.get(target_lang, target_lang)
        numbered = "\n".join(f"{i}. {c.text.replace(chr(10), ' ')}" for i, c in enumerate(cues, start=1))
        system_msg = (
            f"You are a professional subtitle translator. Translate the numbered dialogue lines into {target_name} ({target_lang}).\n"
            f"Output strictly {len(cues)} lines in 'number. translated text' format without Markdown formatting or explanations."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": numbered},
            ],
            "temperature": 0.2,
        }
        req_data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=req_data,
            headers=headers,
            method="POST",
        )

        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8"))
                        choices = body.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            return _parse_lines(content)
                    last_err = f"API returned status {resp.status}"
            except (urllib.error.URLError, TimeoutError) as err:
                last_err = f"API connection error: {err}"
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(last_err or "Failed to translate block with API")


def translate_cues(
    cues: list[Cue],
    target_lang: str,
    client: OllamaClient | OpenAIClient,
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
    provider: str = "ollama",
    model: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    batch_size: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    p = Path(path)
    content, _ = read(p)
    cues = parse_srt(content)
    if not cues:
        raise ValueError(f"No cues found in {p}")

    if provider.lower() in ("openai", "openrouter", "groq", "deepseek"):
        base_url = url or (
            "https://openrouter.ai/api/v1" if provider.lower() == "openrouter" else
            "https://api.groq.com/openai/v1" if provider.lower() == "groq" else
            "https://api.deepseek.com/v1" if provider.lower() == "deepseek" else
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        chosen_model = model or ("gpt-4o-mini" if provider.lower() == "openai" else "llama-3.3-70b-versatile" if provider.lower() == "groq" else "deepseek-chat")
        client = OpenAIClient(api_key=api_key, base_url=base_url, model=chosen_model)
    else:
        chosen_url = url or "http://127.0.0.1:11434"
        chosen_model = model or "gemma3:12b"
        client = OllamaClient(url=chosen_url, model=chosen_model)

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
