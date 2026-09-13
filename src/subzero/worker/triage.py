"""Morning digest for jobs waiting on review. Advisory only: clustering is
rule-based and deterministic, Apple FM only writes the human summary, and a
refusal or outage falls back to a template. Nothing here touches subtitles."""

import re

REVIEW_STATES = ("needs_review", "failed")

STRIP_PATTERNS = (
    re.compile(r"/[^\s:]+"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}[,\.]\d+\b"),
    re.compile(r"\b\d+\.\d+s(?:econds?)?\b"),
    re.compile(r"\b[0-9a-f]{8,}\b"),
    re.compile(r"\b\d+/\d+\b"),
    re.compile(r"\b\d+\b"),
)


def signature(message: str) -> str:
    """Cluster key for a failure: same shape, different ids and numbers."""
    text = (message or "unknown error").lower()
    for pattern in STRIP_PATTERNS:
        text = pattern.sub(" ", text)
    words = [word for word in text.split() if any(char.isalnum() for char in word)]
    if len(words) > 14:
        words = words[:14]
    return " ".join(words) or "unknown error"


def sanitize(message: str) -> str:
    """Example text for display: same stripping as the signature, case kept."""
    text = message or ""
    for pattern in STRIP_PATTERNS:
        text = pattern.sub(" ", text)
    return " ".join(text.split())


def cluster(jobs: list) -> list[dict]:
    """Group review jobs by failure shape. Accepts Job objects or job_json dicts."""
    groups: dict[str, dict] = {}

    def field(job, name):
        return job.get(name) if isinstance(job, dict) else getattr(job, name, "")

    for job in jobs:
        if field(job, "state") not in REVIEW_STATES:
            continue
        key = signature(str(field(job, "message") or field(job, "phase") or ""))
        group = groups.setdefault(key, {"ids": [], "kinds": set(), "example": ""})
        group["ids"].append(str(field(job, "id"))[:8])
        if field(job, "kind"):
            group["kinds"].add(field(job, "kind"))
        if not group["example"]:
            example = sanitize(str(field(job, "message") or field(job, "phase") or ""))
            group["example"] = example[:300]
    ordered = sorted(groups.values(), key=lambda group: -len(group["ids"]))
    for group in ordered:
        group["kinds"] = sorted(group["kinds"])
    return ordered


def template_digest(groups: list[dict]) -> str:
    if not groups:
        return "Review queue is clear."
    lines = [f"{sum(len(g['ids']) for g in groups)} jobs need review:"]
    for group in groups:
        kinds = "/".join(group["kinds"]) or "mixed"
        lines.append(f"- {len(group['ids'])}x [{kinds}] {group['example']}")
    return "\n".join(lines)


SUMMARY_SYSTEM = (
    "Summarize subtitle-worker failures for the operator in plain sentences, "
    "one short paragraph per group: what failed, the likely cause, and the "
    "suggested action. Every group below needs operator review; never conclude "
    "no action is needed. Never invent job ids, counts, or error text.")


def summarize(groups: list[dict], fm) -> str:
    """FM-written digest with a template fallback. FM never sees raw ids."""
    if not groups:
        return "Review queue is clear."
    brief = "\n".join(
        f"Group of {len(g['ids'])} [{'/'.join(g['kinds']) or 'mixed'}]: {g['example']}"
        for g in groups)
    try:
        return fm.generate([{"role": "system", "content": SUMMARY_SYSTEM},
                            {"role": "user", "content": brief}], max_tokens=400)
    except Exception:
        return template_digest(groups)
