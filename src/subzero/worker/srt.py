import re
from dataclasses import dataclass
from typing import Iterator

TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


def seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        head = 0
        if lines[0].strip().isdigit() and len(lines) > 1:
            head = 1
        stamp = TIME.search(lines[head]) if head < len(lines) else None
        if not stamp:
            continue
        body = "\n".join(lines[head + 1:]).strip()
        if not body:
            continue
        g = stamp.groups()
        cues.append(Cue(len(cues) + 1, seconds(*g[:4]), seconds(*g[4:]), body))
    return cues


# marcas de legenda para surdos: som entre colchetes ou parenteses, nome de quem fala
# em maiusculas antes dos dois pontos, tags HTML/ASS e a musiquinha
BRACKETED = re.compile(r"[\[(][^\])]*[\])]")
SPEAKER = re.compile(r"^\s*[-–]?\s*[A-ZÀ-Ú][A-ZÀ-Ú0-9 .'#&/-]{1,24}:\s*")
MUSIC = re.compile(r"[♪♫#\u266a\u266b\u2669\u266c]+")
TAG = re.compile(r"</?[a-zA-Z0-9_-]+[^>]*>")
ASS = re.compile(r"\{[^}]*\}")


def strip_hearing_impaired(cues: list["Cue"], strip_tags: bool = True) -> list["Cue"]:
    """Tira o que so existe para quem nao ouve e devolve a legenda comum.

    Some com [PORTA RANGE], (risos), tags HTML/ASS, notas musicais e o "JOHN:" antes da fala.
    Cue que fica vazia depois disso era so descricao de som e sai fora, com os indices refeitos.
    """
    kept: list[Cue] = []
    for cue in cues:
        lines = []
        for line in cue.text.splitlines():
            if strip_tags:
                line = TAG.sub("", line)
                line = ASS.sub("", line)
            # linha com nota musical e descricao de som inteira, nao fala: sai toda
            if MUSIC.search(line):
                continue
            line = BRACKETED.sub(" ", line)
            line = SPEAKER.sub("", line)
            line = re.sub(r"\s{2,}", " ", line).strip()
            # um travessao sozinho sobra quando a fala dele era so o ruido
            if line in ("-", "–", ""):
                continue
            lines.append(line)
        text = "\n".join(lines).strip()
        if not text:
            continue
        kept.append(Cue(len(kept) + 1, cue.start, cue.end, text))
    return kept


def timestamp(value: float) -> str:
    ms = int(round(value * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def dump(cues: list[Cue]) -> str:
    out = []
    for n, cue in enumerate(cues, start=1):
        out.append(f"{n}\n{timestamp(cue.start)} --> {timestamp(cue.end)}\n{cue.text}\n")
    return "\n".join(out)


def chunks(cues: list[Cue], size: int) -> Iterator[list[Cue]]:
    for i in range(0, len(cues), size):
        yield cues[i:i + size]
