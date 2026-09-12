from __future__ import annotations

import re
from dataclasses import dataclass

from subzero.core import Options, Stats, analyze, fix_text
from .service import same_language

EXCELLENCE_OPTIONS = Options(
    languages=("pt", "en"),
    max_line=42,
    preserve_breaks=True,
    strip_brackets=True,
    strip_parens=True,
    strip_music=True,
    strip_labels=True,
)

UNCLOSED_TAG = re.compile(r"<[a-zA-Z]+(?![^<>\n]*>)")
PT_MT_CAPS = re.compile(r"(?<=[a-z\u00e0-\u00fa,;])\s+(Não|Com|Para|Por|Que)\s+(?=[a-z\u00e0-\u00fa])")


PT_WORDS = {"você", "não", "que", "com", "para", "está", "ele", "ela", "isso", "aqui", "tinha", "muito", "mais", "sobre", "tudo"}
EN_WORDS = {"the", "you", "that", "with", "this", "they", "from", "have", "what", "were", "about", "there", "their", "would"}


def check_language_completeness(text: str, target_lang: str | None) -> tuple[bool, str]:
    if not target_lang or not target_lang.lower().startswith("pt"):
        return True, "pass"
    blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    if len(blocks) < 20:
        return True, "pass"
    tail_len = min(50, max(10, len(blocks) // 5))
    tail_text = " ".join(blocks[-tail_len:]).lower()
    words = re.findall(r"\b[a-zà-ú]+\b", tail_text)
    en_hits = sum(1 for w in words if w in EN_WORDS)
    pt_hits = sum(1 for w in words if w in PT_WORDS)
    if en_hits >= 15 and pt_hits < 3:
        return False, f"Falha no guard de traducao: final da legenda nao traduzido ({en_hits} termos em ingles vs {pt_hits} em portugues)"
    return True, "pass"


@dataclass(frozen=True)
class GuardReport:
    ok: bool
    reason: str
    stats: Stats | None = None


def check_excellence_guards(
    text: str,
    target_lang: str | None = None,
    accepted_langs: tuple[str, ...] | list[str] | None = None,
    opts: Options = EXCELLENCE_OPTIONS,
) -> GuardReport:
    """Valida os guards de excelencia do worker:
    1. Idioma alvo: se configurado, valida contra a lista de idiomas permitidos.
    2. Sem SDH: zero marcacoes sonoras, notas musicais ou rotulos de locutor.
    3. Sem colisoes: dialogos com multiplos locutores devidamente separados.
    4. Formatacao e limites: linhas dentro do limite maximo de caracteres e tags validas.
    5. Qualidade de traducao: deteccao de artefatos grosseiros e completude da traducao.
    6. Estrutura integra: arquivo nao vazio com blocos de tempo validos.
    """
    if accepted_langs and target_lang:
        if not any(same_language(target_lang, al) for al in accepted_langs):
            return GuardReport(False, f"Idioma nao aceito pelo worker: esperado {accepted_langs}, recebido {target_lang}")
    if not text or not text.strip():
        return GuardReport(False, "Legenda vazia")
    if UNCLOSED_TAG.search(text):
        return GuardReport(False, "Falha no guard de formatacao: contem tags HTML malformadas ou nao fechadas")
    if target_lang and target_lang.lower().startswith("pt"):
        mt_matches = PT_MT_CAPS.findall(text)
        if mt_matches:
            return GuardReport(False, f"Falha no guard de traducao: contem {len(mt_matches)} particulas capitalizadas no meio da frase")
        lang_ok, lang_reason = check_language_completeness(text, target_lang)
        if not lang_ok:
            return GuardReport(False, lang_reason)
    try:
        st = analyze(text, opts)
    except Exception as err:
        return GuardReport(False, f"Erro ao analisar estrutura da legenda: {err}")
    if not st.cues:
        return GuardReport(False, "Legenda sem blocos de tempo validos", st)
    if st.sdh > 0:
        return GuardReport(False, f"Falha no guard de SDH: contem {st.sdh} termos/indicadores SDH", st)
    if st.collapsed > 0:
        return GuardReport(False, f"Falha no guard de dialogo: contem {st.collapsed} falas colididas", st)
    if st.long_lines > 0:
        return GuardReport(False, f"Falha no guard de formatacao: contem {st.long_lines} linhas excessivamente longas", st)
    return GuardReport(True, "pass", st)


PT_MT_FIX = re.compile(r"(?<=[a-z\u00e0-\u00fa,;])(\s+)(Não|Com|Para|Por|Que)(\s+)(?=[a-z\u00e0-\u00fa])")


def sanitize_to_excellence(
    text: str,
    target_lang: str | None = None,
    accepted_langs: tuple[str, ...] | list[str] | None = None,
    opts: Options = EXCELLENCE_OPTIONS,
) -> str:
    """Aplica limpeza profunda de termos SDH, correcao de colisoes e quebras de linha."""
    if not text or not text.strip():
        return ""
    guard = check_excellence_guards(text, target_lang=target_lang, accepted_langs=accepted_langs, opts=opts)
    if guard.ok:
        return text
    text = UNCLOSED_TAG.sub("", text)
    if target_lang and target_lang.lower().startswith("pt"):
        text = PT_MT_FIX.sub(lambda m: f"{m.group(1)}{m.group(2).lower()}{m.group(3)}", text)
    result = fix_text(text, opts)
    out = result.text.replace("\r\n", "\n")
    return out.rstrip("\n") + "\n"
