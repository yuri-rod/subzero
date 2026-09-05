from .jobs import Job

KIND_VERB = {"embedded": "extracted", "opensubtitles": "downloaded", "whisper": "transcribed",
            "translate": "translated"}


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

    def job_finished(self, job: Job, item_name: str | None = None) -> None:
        if not self.enabled:
            return
        label = item_name or job.item_id
        ok = job.state == "done"
        verb = KIND_VERB.get(job.kind, job.kind)
        title = "Subtitle ready" if ok else "Subtitle job failed"
        body = (f"{label}: {verb} to {job.target_lang}" if ok
               else f"{label}: {job.kind} -> {job.target_lang} failed ({job.message or 'unknown error'})")
        try:
            self.http.request("POST", f"{self.url}/{self.topic}", content=body.encode("utf-8"),
                              headers={"Title": title, "Priority": "3" if ok else "4",
                                       "Tags": "captions" if ok else "warning"})
        except Exception:
            pass
