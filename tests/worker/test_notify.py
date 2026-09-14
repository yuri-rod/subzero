from subzero.worker.jobs import Job
from subzero.worker.notify import Notifier


class FakeHTTP:
    def __init__(self, raises=None):
        self.calls = []
        self.raises = raises

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.raises:
            raise self.raises
        return type("R", (), {"status_code": 200})()


def job(state="done", kind="whisper", target="pt-BR", message="", origin="auto",
        attempts=0, job_id="j1abcdef"):
    return Job(id=job_id, item_id="abc", kind=kind, target_lang=target, source_id=None,
              origin=origin, state=state, phase="", percent=100 if state == "done" else 0,
              message=message, result_path=None, attempts=attempts, next_attempt=0,
              created=0, updated=0)


def test_disabled_without_a_topic():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "", http=http).job_finished(job(origin="manual"))
    assert http.calls == []


def test_done_job_posts_a_success_message():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(origin="manual"), item_name="Filme")

    method, url, kwargs = http.calls[0]
    assert url == "https://ntfy.sh/yuri-topic"
    assert b"Filme" in kwargs["content"]
    assert kwargs["headers"]["Title"] == "Subtitle ready"


def test_auto_done_stays_quiet():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(state="done", origin="auto"), item_name="Filme")

    assert http.calls == []


def test_failed_job_posts_a_higher_priority_warning():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(state="failed", message="ollama caiu"), item_name="Filme")

    _, _, kwargs = http.calls[0]
    assert kwargs["headers"]["Title"] == "Subtitle job failed"
    assert kwargs["headers"]["Priority"] == "4"
    assert b"ollama caiu" in kwargs["content"]


def test_failed_job_reports_retry_count():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(state="failed", message="timeout", attempts=5), item_name="Filme")

    _, _, kwargs = http.calls[0]
    assert b"after 5 retries" in kwargs["content"]


def test_needs_review_has_its_own_title():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(state="needs_review", kind="repair", message="caption lost"), item_name="Filme")

    _, _, kwargs = http.calls[0]
    assert kwargs["headers"]["Title"] == "Subtitle needs review"
    assert kwargs["headers"]["Priority"] == "4"
    assert kwargs["headers"]["Tags"] == "eyes"
    assert b"caption lost" in kwargs["content"]
    assert b"j1abcdef" in kwargs["content"]


def test_paused_job_holds_the_queue_at_urgent_priority():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(state="paused", kind="repair", message="120 chars remain"), item_name="Filme")

    _, _, kwargs = http.calls[0]
    assert kwargs["headers"]["Title"] == "Queue paused"
    assert kwargs["headers"]["Priority"] == "5"
    assert b"120 chars remain" in kwargs["content"]
    assert b"resume job j1abcdef" in kwargs["content"]


def test_rebuild_done_uses_a_past_verb():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(state="done", kind="rebuild", origin="manual"), item_name="Filme")

    _, _, kwargs = http.calls[0]
    assert b"rebuilt to pt-BR" in kwargs["content"]


def test_sweep_posts_a_low_priority_summary():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).sweep(
        [job(kind="opensubtitles"), job(kind="translate"), job(kind="translate")],
        downloads_today=12, budget=15)

    _, url, kwargs = http.calls[0]
    assert url == "https://ntfy.sh/yuri-topic"
    assert kwargs["headers"]["Title"] == "Auto sweep"
    assert kwargs["headers"]["Priority"] == "2"
    assert b"Enqueued 3 jobs (downloads today 12/15)" in kwargs["content"]
    assert b"1 opensubtitles, 2 translate" in kwargs["content"]


def test_sweep_stays_quiet_when_empty_or_disabled():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).sweep([])
    Notifier("https://ntfy.sh", "", http=http).sweep([job()])

    assert http.calls == []


def test_a_network_error_never_raises():
    Notifier("https://ntfy.sh", "yuri-topic",
             http=FakeHTTP(raises=ConnectionError())).job_finished(job(state="failed"))
    Notifier("https://ntfy.sh", "yuri-topic",
             http=FakeHTTP(raises=ConnectionError())).sweep([job()])
