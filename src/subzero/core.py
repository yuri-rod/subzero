"""Strip SDH markup from SRT subtitles and repair their line breaks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace

from .roles import words_for

TIMECODE = r"\d\d:\d\d:\d\d(?:[,.]\d+)?"
CUE = re.compile(rf"({TIMECODE}\s*-->\s*{TIMECODE}[^\n]*)\s*\n(.*?)(?=\n\s*\n|\Z)", re.S)
ASS = re.compile(r"\{[^}]*\}")
TAG = re.compile(r"</?[a-z][^>]*>", re.I)
MUSIC = re.compile(r"[\u266a\u266b\u2669\u266c]")
CAPS_LABEL = r"[A-Z\u00c0-\u00dc\u00c7][A-Z\u00c0-\u00dc\u00c70-9 .'#\-]{1,20}"
# a second speaker's dash, with or without the space some releases omit
SPEAKER_DASH = re.compile(r"\S\s+-\s*\S")
PUNCT_DASH = re.compile(r'[.!?…:\u2026"\'\u201d\u2019]\s*-\s*\S')
SPLIT_DIALOGUE = re.compile(
    r"(?<=[.!?…:\u2026\"\'\u201d\u2019])\s*-\s*(?=\S)"
    rf"|\s+-\s*(?={CAPS_LABEL}:)"
    r"|\s+-\s*(?=\S)"
)
SPLIT_DASH = re.compile(r"\s+-\s*(?=\S)")
BREAK_AFTER = re.compile(r"[.,;:!?\u2026]$")
CONTINUES = re.compile(
    r"(?i)^(e|ou|mas|que|de|do|da|em|no|na|para|pra|com|por|se|um|uma|os|as"
    r"|and|or|but|that|of|the|to|in|on|at|for|with|a|an)\b"
)


@dataclass(frozen=True)
class Options:
    """Knobs for a run. Defaults match common subtitle practice."""

    max_line: int = 42
    languages: tuple[str, ...] = ("en", "pt")
    preserve_breaks: bool = True
    strip_brackets: bool = True
    strip_parens: bool = True
    strip_music: bool = True
    strip_labels: bool = True

    def __post_init__(self):
        if isinstance(self.languages, (list, set)):
            object.__setattr__(self, "languages", tuple(self.languages))


@dataclass
class Result:
    text: str
    cues: int
    dropped: int
    rewrapped: int
    changed: bool = False


@dataclass
class Stats:
    """What a file looks like before anything is done to it."""

    cues: int = 0
    encoding: str = "utf-8"
    sdh: int = 0
    collapsed: int = 0
    long_lines: int = 0
    multiline: int = 0
    empty: int = 0

    def pct(self, name: str) -> float:
        return round(100.0 * getattr(self, name) / self.cues, 1) if self.cues else 0.0


@dataclass(frozen=True)
class _Rules:
    label: re.Pattern
    spans: re.Pattern | None


_CACHE: dict[Options, _Rules] = {}


def rules_for(opts: Options) -> _Rules:
    """Compile the patterns an Options implies, once per distinct Options."""
    if opts in _CACHE:
        return _CACHE[opts]
    roles = "|".join(re.escape(w) for w in words_for(opts.languages))
    parts = [CAPS_LABEL]
    if roles:
        # a role optionally followed by a qualifier: "man 2:", "voz masculina:"
        parts.append(rf"(?:{roles})(?:\s+(?:[a-z\u00e0-\u00ff]+|\d+)){{0,2}}")
    label = re.compile(rf"^\s*(?:{'|'.join(parts)}):(?!\d)(?:\s+|$)")
    spans = []
    if opts.strip_brackets:
        spans.append(r"\[[^\]]*\]")
    if opts.strip_parens:
        spans.append(r"\([^)]*\)")
    rules = _Rules(label=label, spans=re.compile("|".join(spans)) if spans else None)
    _CACHE[opts] = rules
    return rules


def _tag_split(text: str) -> tuple[str, str, str]:
    """Peel a tag that wraps the whole cue, so a break never lands inside it."""
    m = re.fullmatch(r"\s*(<([a-z]+)[^>]*>)(.*?)(</\2>)\s*", text, re.S | re.I)
    if not m:
        return ("", text, "")
    open_tag, tag_name, inner, close_tag = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
    first_close = re.search(rf"</{tag_name}>", inner, re.I)
    if first_close:
        first_open = re.search(rf"<{tag_name}\b[^>]*>", inner, re.I)
        if not first_open or first_close.start() < first_open.start():
            return ("", text, "")
    return (open_tag, inner, close_tag)


def is_collapsed(line: str) -> bool:
    line = TAG.sub("", line).strip()
    if not line:
        return False
    if line.startswith("-"):
        rest = line.lstrip("- ")
        return bool(SPEAKER_DASH.search(rest) or PUNCT_DASH.search(rest) or re.search(rf"\s-\s*{CAPS_LABEL}:", rest))
    return bool(PUNCT_DASH.search(line) or re.search(rf"\s-\s*{CAPS_LABEL}:", line))


def balance(text: str, limit: int) -> list[str]:
    """Break one run of text at the best point near its middle.

    Punctuation beats a bare gap: a cut after a comma reads better than one that
    merely happens to fall in the centre.
    """
    def clean_len(s: str) -> int:
        return len(TAG.sub("", s).strip())

    if clean_len(text) <= limit or " " not in text:
        return [text]
    mid = len(text) / 2
    best = None
    for m in re.finditer(r"\s+", text):
        left, right = text[: m.start()].rstrip(), text[m.end():].lstrip()
        if not left or not right:
            continue
        c_left, c_right = clean_len(left), clean_len(right)
        cost = abs(m.start() - mid)
        if BREAK_AFTER.search(left):
            cost -= 12
        elif CONTINUES.match(right):
            cost -= 5
        if c_left > limit:
            cost += (c_left - limit) * 8
        if c_right > limit:
            cost += (c_right - limit) * 8
        if best is None or cost < best[0]:
            best = (cost, left, right)
    if best is None:
        return [text]
    _, left, right = best
    out = []
    if clean_len(left) > limit and " " in left:
        out.extend(balance(left, limit))
    else:
        out.append(left)
    if clean_len(right) > limit and " " in right:
        out.extend(balance(right, limit))
    else:
        out.append(right)
    return out


def strip_sdh(body: str, opts: Options) -> str:
    """Remove sound cues and music marks. Empty means the cue was only noise."""
    rules = rules_for(opts)
    body = ASS.sub("", body)
    if rules.spans:
        body = rules.spans.sub(" ", body)
    if opts.strip_music:
        body = MUSIC.sub(" ", body)
    lines = []
    for line in body.split("\n"):
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line in {"-", "--", "...", ".", ",", "!", "?"}:
            continue
        line = re.sub(r"^-\s*(?=[.,!?;:])", "", line)
        if TAG.sub("", line).strip(" -.,!?;:"):
            lines.append(line)
    return "\n".join(lines).strip()


def strip_label(part: str, opts: Options) -> str:
    """Drop a speaker label, after the speakers have been separated.

    Before separation the second speaker's label sits mid-line, where an anchored
    pattern can never reach it.
    """
    if not opts.strip_labels:
        return part.strip()
    lead = "- " if part.lstrip().startswith("-") else ""
    body = part.lstrip()[1:].strip() if lead else part
    body = rules_for(opts).label.sub("", body, count=1)
    return (lead + body).strip() if body else ""


def rewrap(text: str, opts: Options) -> str:
    """Rebuild a cue's line breaks from scratch."""
    open_tag, body, close_tag = _tag_split(text)
    body = " ".join(l.strip() for l in body.split("\n") if l.strip())
    if not body:
        return ""
    body_clean = TAG.sub("", body).strip()
    is_dialogue = (
        is_collapsed(body_clean)
        or bool(rules_for(opts).label.match(body_clean))
        or bool(re.search(rf"\s-\s*{CAPS_LABEL}:", body_clean))
    )
    parts = None
    if is_dialogue:
        content = body.strip()
        if content.startswith("-"):
            content = content.lstrip("- ")
        chunks = [c.strip() for c in SPLIT_DIALOGUE.split(content) if c.strip()]
        if len(chunks) > 1:
            parts = ["- " + strip_label(c, opts) for c in chunks]
        elif chunks:
            parts = ["- " + strip_label(chunks[0], opts)]
    if parts is None:
        parts = [strip_label(body, opts)]
    parts = [p for p in parts if p.strip(" -")]
    if not parts:
        return ""
    if len(parts) == 1:
        one = parts[0]
        if one.startswith("- "):
            one = one[2:]
        parts = balance(one, opts.max_line)
    else:
        out = []
        for p in parts:
            if len(TAG.sub("", p).strip()) > opts.max_line:
                lead = "- " if p.startswith("- ") else ""
                clean_p = p[2:] if lead else p
                balanced = balance(clean_p, opts.max_line)
                if lead and balanced:
                    balanced[0] = lead + balanced[0]
                out.extend(balanced)
            else:
                out.append(p)
        parts = out
    joined = "\n".join(p for p in parts if p)
    return f"{open_tag}{joined}{close_tag}" if open_tag else joined


def keep_breaks(text: str, opts: Options) -> str:
    """Leave a cue's existing breaks alone, removing only the label.

    Subtitles that shipped with a release are usually broken by a human and broken
    well. Rewrapping those joins the lines and re-splits them by length, undoing
    work that was already right. Only a clearly broken cue is rebuilt: a line over
    the limit, or two speakers stuck together.
    """
    open_tag, body, close_tag = _tag_split(text)
    lines = []
    for line in body.split("\n"):
        line = strip_label(line.strip(), opts)
        if line.strip(" -"):
            lines.append(line)
    if not lines:
        return ""
    broken = any(len(TAG.sub("", l).strip()) > opts.max_line for l in lines) or any(
        is_collapsed(l) for l in lines
    )
    if broken:
        return rewrap(text, opts)
    if len(lines) == 1:
        # a lone dash on a one-line cue usually marks that the exchange carries
        # over from the cue before, so it is meaningful and stays
        joined = lines[0]
        return f"{open_tag}{joined}{close_tag}" if open_tag else joined
    joined = "\n".join(lines)
    return f"{open_tag}{joined}{close_tag}" if open_tag else joined


def normalise(raw: str) -> str:
    cleaned = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace(chr(8212), "-").replace(chr(8211), "-")
    return cleaned


def fix_text(raw: str, opts: Options | None = None) -> Result:
    """Clean every cue, drop the ones that were pure SDH, renumber the rest."""
    opts = opts or Options()
    raw = normalise(raw)
    kept: list[tuple[str, str]] = []
    dropped = rewrapped = 0
    for timing, body in CUE.findall(raw):
        body = body.strip()
        cleaned = strip_sdh(body, opts)
        if not cleaned:
            dropped += 1
            continue
        text = keep_breaks(cleaned, opts) if opts.preserve_breaks else rewrap(cleaned, opts)
        if not text:
            dropped += 1
            continue
        if text != body:
            rewrapped += 1
        kept.append((timing.strip(), text))
    out = "".join(f"{i}\n{t}\n{x}\n\n" for i, (t, x) in enumerate(kept, 1))
    return Result(text=out.replace("\n", "\r\n"), cues=len(kept),
                  dropped=dropped, rewrapped=rewrapped,
                  changed=bool(dropped or rewrapped))


def analyze(raw: str, opts: Options | None = None) -> Stats:
    """Describe a file without changing it, for the check command."""
    opts = opts or Options()
    rules = rules_for(opts)
    raw = normalise(raw)
    st = Stats()
    for _, body in CUE.findall(raw):
        body = body.strip()
        st.cues += 1
        lines = [l for l in body.split("\n") if l.strip()]
        if len(lines) > 1:
            st.multiline += 1
        if any(is_collapsed(l) for l in lines):
            st.collapsed += 1
        if any(len(TAG.sub("", l).strip()) > opts.max_line + 3 for l in lines):
            st.long_lines += 1
        if "[" in body or "(" in body or MUSIC.search(body) or rules.label.match(body):
            st.sdh += 1
        if not strip_sdh(body, opts):
            st.empty += 1
    return st


def read(path) -> tuple[str, str]:
    """Read a subtitle, tolerating legacy 8-bit and unicode encodings still in the wild."""
    data = open(path, "rb").read()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1", "iso-8859-1", "windows-1250"):
        try:
            return data.decode(enc), ("utf-8" if "utf-8" in enc else enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("subzero", data, 0, 1, "no supported encoding")


def fix_file(path, opts: Options | None = None, backup_dir=None,
             dry: bool = False) -> Result | None:
    """Fix one file in place. Returns None when nothing parsed as a cue."""
    raw, encoding = read(path)
    result = fix_text(raw, opts)
    if not result.cues:
        return None
    result.changed = result.changed or encoding != "utf-8"
    if dry or not result.changed:
        return result
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        dest = os.path.join(backup_dir, os.path.basename(str(path)))
        # never overwrite an existing backup: the first one is the true original
        if not os.path.exists(dest):
            with open(dest, "wb") as fh:
                fh.write(open(path, "rb").read())
    tmp = f"{path}.subzero-part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(result.text)
    os.replace(tmp, path)
    return result
