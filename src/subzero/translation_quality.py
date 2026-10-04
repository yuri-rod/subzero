"""Semantic quality signal for translated subtitles.

Deterministic tripwires catch gross defects without a model; a local Ollama
judge scores a sample of cue pairs for meaning preservation. The judge is
best-effort: any failure degrades to the deterministic-only verdict instead
of failing the job.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

from .translate import GAME_TERMS, MAX_RESPONSE_BYTES

SAMPLE_CUES = 40
JUDGE_BATCH = 20
JUDGE_TIMEOUT = 180
REVIEW_MEAN = 3.5
REVIEW_CRITICALS = 2
REVIEW_DET_HITS = 5

_TAG = re.compile(r"</?[a-z][^>]*>|\{[^}]*\}")


class JudgeUnavailable(RuntimeError):
    """The LLM judge could not score; fall back to deterministic tripwires."""


@dataclass(frozen=True)
class Judgment:
    cue: int  # 0-based index into the scored cue lists
    adequacy: int  # 1-5 meaning preservation score
    critical: bool  # inverted/negated meaning or wrong entity
    reason: str


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    score: float | None = None
    worst: tuple[Judgment, ...] = ()


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub("", text or "")).strip()


def pair_hits(source: str, translated: str) -> list[str]:
    """Deterministic tripwires for one source/translation pair."""
    src, tgt = _plain(source), _plain(translated)
    if not src:
        return []
    if not tgt:
        return ["empty translation"]
    hits = []
    long, short = max(len(src), len(tgt)), min(len(src), len(tgt))
    if short and long / short > 3 and long - short > 20:
        hits.append("extreme length ratio")
    match = GAME_TERMS.search(src)
    if match:
        term = match.group(0).lower()
        low = tgt.casefold()
        if "immunity" in term and "ídolo" not in low and "idolo" not in low:
            hits.append("game term 'immunity idol' mistranslated")
        if "tribal" in term and "conselho" not in low:
            hits.append("game term 'tribal council' mistranslated")
    return hits


def deterministic_hits(sources: list[str], translated: list[str]) -> dict[int, list[str]]:
    """Map cue index to deterministic defect reasons, including file-level repeats."""
    hits: dict[int, list[str]] = {}
    for n, (src, tgt) in enumerate(zip(sources, translated)):
        found = pair_hits(src, tgt)
        if found:
            hits[n] = found
    groups: dict[str, list[int]] = {}
    for n, tgt in enumerate(translated):
        groups.setdefault(_plain(tgt).casefold(), []).append(n)
    for key, idxs in groups.items():
        if len(key) >= 20 and len({sources[n] for n in idxs}) >= 3:
            for n in idxs:
                hits.setdefault(n, []).append("same translation repeated for distinct dialogue")
    return hits


def sample_indexes(count: int, suspects=(), n: int = SAMPLE_CUES) -> list[int]:
    """Even stride across the file with every suspect cue kept, capped at n."""
    if count <= 0:
        return []
    if count <= n:
        return list(range(count))
    stride = [(i * count) // n for i in range(n)]
    keep = list(dict.fromkeys([s for s in suspects if 0 <= s < count] + stride))
    return sorted(keep[:n])


_JUDGE_INTRO = (
    "Rate each Brazilian Portuguese subtitle translation for meaning preservation.\n"
    "Score adequacy 1-5: 5 same meaning and natural, 3 understandable with issues, "
    "1 wrong or unrelated. Mark critical true when meaning is inverted or negated, "
    "or the wrong person or thing is named.\n"
    "Reply with a JSON array of exactly {n} objects with integer id, integer adequacy "
    "1-5, boolean critical, and short string reason.\n"
    "PAIRS:\n"
)


def _judge_payload(pairs: list[tuple[int, str, str]], model: str) -> dict:
    numbered = json.dumps(
        [{"id": n, "en": src, "pt": tgt} for n, src, tgt in pairs], ensure_ascii=False
    )
    return {
        "model": model,
        "prompt": _JUDGE_INTRO.format(n=len(pairs)) + numbered,
        "format": {
            "type": "array",
            "minItems": len(pairs),
            "maxItems": len(pairs),
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "adequacy": {"type": "integer", "minimum": 1, "maximum": 5},
                    "critical": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "adequacy", "critical", "reason"],
                "additionalProperties": False,
            },
        },
        "stream": False,
        "options": {"temperature": 0},
    }


def parse_judgments(ids: list[int], body: dict) -> list[Judgment]:
    """Parse one judge batch; anything unexpected means the judge is unusable."""
    try:
        raw = body.get("response")
        items = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        items = None
    if not isinstance(items, list):
        raise JudgeUnavailable("judge returned no JSON array")
    wanted = set(ids)
    by_id: dict[int, Judgment] = {}
    for item in items:
        if not isinstance(item, dict):
            raise JudgeUnavailable("judge returned a malformed entry")
        cue, adequacy = item.get("id"), item.get("adequacy")
        critical, reason = item.get("critical"), item.get("reason")
        if (
            not isinstance(cue, int)
            or isinstance(cue, bool)
            or cue not in wanted
            or not isinstance(adequacy, int)
            or isinstance(adequacy, bool)
            or not 1 <= adequacy <= 5
            or not isinstance(critical, bool)
            or not isinstance(reason, str)
        ):
            raise JudgeUnavailable("judge returned a malformed entry")
        by_id[cue] = Judgment(cue, adequacy, critical, reason.strip()[:200])
    if set(by_id) != wanted:
        raise JudgeUnavailable("judge skipped cues")
    return [by_id[n] for n in ids]


def _judge_batch(
    batch: list[tuple[int, str, str]], *, url: str, model: str, timeout: int
) -> list[Judgment]:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(_judge_payload(batch, model)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, ValueError) as err:
        raise JudgeUnavailable(f"judge request failed: {err}") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise JudgeUnavailable("judge response too large")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise JudgeUnavailable("judge response is not JSON") from None
    if not isinstance(body, dict):
        raise JudgeUnavailable("judge response is not JSON")
    return parse_judgments([n for n, _, _ in batch], body)


def judge_pairs(
    pairs: list[tuple[int, str, str]], *, url: str, model: str, timeout: int = JUDGE_TIMEOUT
) -> list[Judgment]:
    """Score sampled pairs with the local Ollama judge, 20 pairs per request."""
    out: list[Judgment] = []
    for start in range(0, len(pairs), JUDGE_BATCH):
        out.extend(_judge_batch(pairs[start : start + JUDGE_BATCH], url=url, model=model, timeout=timeout))
    return out


def check_translation(sources: list[str], translated: list[str], *, url: str, model: str) -> Verdict:
    """Score a translated cue list against its source; judge outage degrades, never fails."""
    if len(sources) != len(translated):
        raise ValueError("source and translation cue counts differ")
    if not sources:
        return Verdict(True, "no cues to judge", None, ())
    det = deterministic_hits(sources, translated)
    pairs = [(n, sources[n], translated[n]) for n in sample_indexes(len(sources), sorted(det))]
    try:
        judgments = judge_pairs(pairs, url=url, model=model)
    except JudgeUnavailable as err:
        if len(det) >= REVIEW_DET_HITS:
            detail = "; ".join(f"#{n + 1}: {det[n][0]}" for n in sorted(det)[:5])
            return Verdict(False, f"judge unavailable ({err}); {len(det)} suspect cues: {detail}", None, ())
        return Verdict(True, f"judge unavailable ({err}); {len(det)} suspect cues", None, ())
    score = sum(j.adequacy for j in judgments) / len(judgments)
    criticals = [j for j in judgments if j.critical]
    worst = tuple(sorted(judgments, key=lambda j: (j.adequacy, j.cue))[:5])
    if score < REVIEW_MEAN or len(criticals) >= REVIEW_CRITICALS or len(det) >= REVIEW_DET_HITS:
        detail = "; ".join(f"#{j.cue + 1} ({j.adequacy}/5): {j.reason}" for j in worst)
        return Verdict(
            False,
            f"score {score:.1f}/5 over {len(judgments)} cues, {len(criticals)} critical: {detail}",
            score,
            worst,
        )
    return Verdict(True, f"score {score:.1f}/5 over {len(judgments)} cues", score, worst)
