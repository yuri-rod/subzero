from __future__ import annotations

import re
from dataclasses import dataclass

from subzero.convert import parse_srt
from subzero.core import Options, Stats, analyze, fix_text
from .service import same_language
from .srt import ASS, TAG

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


PT_WORDS = set("""
    eu você vocês voce voces não nao que com para está esta ele ela eles elas
    isso isto aqui tinha muito muita mais sobre tudo nós nos uma um por porque
    mas sim sei meu minha seu sua também tambem são sao é até então ainda
    agora vai vou vamos foi era estava estou estamos somos quero pode fazer
    e o os as de da das do dos na no nas nos em se já pra ao à às né só
""".split())
EN_WORDS = set("""
    the you your yours that this these those they them their theirs from
    have has had what were was where when why how there would should could
    can cannot will shall must might be been being am are is it its we our
    ours he his she her hers and but if then than because which who whom
    with without about into through between before after under over while
    never always every not don't doesn't didn't isn't aren't wasn't weren't
    won't wouldn't shouldn't couldn't can't i'm you're he's she's it's we're
    they're i've you've we've they've i'll you'll he'll she'll we'll they'll
    that's there's let's i my myself themselves ourselves yourself himself
    herself get got want need think know thought found feel like lost hurt
    much many very really something anything nothing everything someone
    anyone everyone nobody first second third fourth last next one two three
    four five key keys win wins winning
""".split())
LANGUAGE_WORD = re.compile(r"[a-zà-öø-ÿ]+(?:'[a-z]+)?")
EN_CLAUSE = re.compile(
    r"\b(?:(?:i|you|he|she|it|we|they)\s+"
    r"(?:am|is|are|was|were|have|has|had|will|would|can|could|should|must)"
    r"|(?:i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've))"
    r"\s+[a-z]+\b"
)
EN_GAME_TERMS = re.compile(r"\b(?:immunity\s+idols?|tribal\s+councils?)\b")


def check_language_completeness(text: str, target_lang: str | None) -> tuple[bool, str]:
    if not target_lang or not same_language(target_lang.lower().replace("_", "-").split("-")[0], "pt"):
        return True, "pass"
    for index, cue in enumerate(parse_srt(text), start=1):
        dialogue = ASS.sub(" ", TAG.sub("", cue.text)).lower().replace("’", "'")
        reason = (
            "Falha no guard de traducao: trecho nao traduzido "
            f"no bloco {index} em {cue.start}"
        )
        if EN_CLAUSE.search(dialogue) or EN_GAME_TERMS.search(dialogue):
            return False, reason
        words = LANGUAGE_WORD.findall(dialogue)
        for start in range(len(words)):
            english = set()
            for word in words[start:start + 8]:
                if word in PT_WORDS:
                    break
                if word in EN_WORDS:
                    english.add(word)
                if len(english) >= 3:
                    return False, reason
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
    if target_lang and same_language(target_lang.lower().replace("_", "-").split("-")[0], "pt"):
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
    if target_lang and same_language(target_lang.lower().replace("_", "-").split("-")[0], "pt"):
        text = PT_MT_FIX.sub(lambda m: f"{m.group(1)}{m.group(2).lower()}{m.group(3)}", text)
    result = fix_text(text, opts)
    out = result.text.replace("\r\n", "\n")
    return out.rstrip("\n") + "\n"
