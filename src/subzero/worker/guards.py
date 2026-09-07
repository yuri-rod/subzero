from __future__ import annotations

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


@dataclass(frozen=True)
class GuardReport:
    ok: bool
    reason: str
    stats: Stats | None = None


def check_excellence_guards(text: str, target_lang: str, opts: Options = EXCELLENCE_OPTIONS) -> GuardReport:
    """Valida os guards de excelencia do worker:
    1. Idioma alvo: estritamente pt-BR (ou codigo equivalente 'por').
    2. Sem SDH: zero marcacoes sonoras, notas musicais ou rotulos de locutor.
    3. Sem colisoes: dialogos com multiplos locutores devidamente separados.
    4. Formatacao e limites: linhas dentro do limite maximo de caracteres.
    5. Estrutura integra: arquivo nao vazio com blocos de tempo validos.
    """
    if not same_language(target_lang, "pt-BR"):
        return GuardReport(False, f"Idioma nao aceito pelo worker: esperado pt-BR, recebido {target_lang}")
    if not text or not text.strip():
        return GuardReport(False, "Legenda vazia")
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


def sanitize_to_excellence(text: str, opts: Options = EXCELLENCE_OPTIONS) -> str:
    """Aplica limpeza profunda de termos SDH, correcao de colisoes e quebras de linha."""
    if not text or not text.strip():
        return ""
    guard = check_excellence_guards(text, "pt-BR", opts)
    if guard.ok:
        return text
    result = fix_text(text, opts)
    out = result.text.replace("\r\n", "\n")
    return out.rstrip("\n") + "\n"
