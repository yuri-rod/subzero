from .jobs import Job

KIND_VERB = {"embedded": "extracted", "opensubtitles": "downloaded", "whisper": "transcribed",
            "translate": "translated", "embedded_translate": "translated", "audit": "audited",
            "refetch": "refetched", "resync": "resynced", "rebuild": "rebuilt",
            "recover_gaps": "recovered", "repair": "repaired"}


class Notifier:
    """Push para o ntfy quando um job termina. Best-effort: nunca derruba a fila."""

    def __init__(self, url: str, topic: str, http=None):
        self.url = url.rstrip("/")
        self.topic = topic
        if http is None:
            import httpx
            http = httpx.Client(timeout=10)
        self.http = http

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    def digest(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        try:
            self.http.request("POST", f"{self.url}/{self.topic}", content=text.encode("utf-8"),
                              headers={"Title": "Worker triage", "Priority": "3",
                                       "Tags": "clipboard"})
        except Exception:
            pass

    def job_finished(self, job: Job, item_name: str | None = None) -> None:
        if not self.enabled:
            return
        # rotina automatica so apita quando precisa de gente; o sucesso do lote
        # ja foi anunciado pelo sweep que o enfileirou
        if job.state == "done" and job.origin == "auto":
            return
        label = item_name or job.item_id
        short = job.id[:8]
        if job.state == "done":
            verb = KIND_VERB.get(job.kind, job.kind)
            title, priority, tags = "Subtitle ready", "3", "captions"
            body = f"{label}: {verb} to {job.target_lang}"
        elif job.state == "needs_review":
            title, priority, tags = "Subtitle needs review", "4", "eyes"
            body = (f"{label}: {job.kind} -> {job.target_lang} needs review "
                    f"({job.message or 'unknown reason'}) [{short}]")
        elif job.state == "paused":
            title, priority, tags = "Queue paused", "5", "no_entry"
            body = (f"{label}: {job.kind} -> {job.target_lang} paused "
                    f"({job.message or 'quota hold'}). Queue held; "
                    f"resume job {short} when clear")
        else:
            title, priority, tags = "Subtitle job failed", "4", "warning"
            retry = f" after {job.attempts} retries" if job.attempts else ""
            body = (f"{label}: {job.kind} -> {job.target_lang} failed{retry} "
                    f"({job.message or 'unknown error'}) [{short}]")
        try:
            self.http.request("POST", f"{self.url}/{self.topic}", content=body.encode("utf-8"),
                              headers={"Title": title, "Priority": priority, "Tags": tags})
        except Exception:
            pass

    def sweep(self, enqueued) -> None:
        jobs = list(enqueued)
        if not self.enabled or not jobs:
            return
        kinds: dict[str, int] = {}
        for job in jobs:
            kinds[job.kind] = kinds.get(job.kind, 0) + 1
        mix = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
        body = f"Enqueued {len(jobs)} jobs: {mix}"
        try:
            self.http.request("POST", f"{self.url}/{self.topic}", content=body.encode("utf-8"),
                              headers={"Title": "Auto sweep", "Priority": "2", "Tags": "radar"})
        except Exception:
            pass
