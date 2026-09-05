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


def job(state="done", kind="whisper", target="pt-BR", message=""):
    return Job(id="j1", item_id="abc", kind=kind, target_lang=target, source_id=None,
              origin="auto", state=state, phase="", percent=100 if state == "done" else 0,
              message=message, result_path=None, attempts=0, next_attempt=0, created=0, updated=0)


def test_disabled_without_a_topic():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "", http=http).job_finished(job())
    assert http.calls == []


def test_done_job_posts_a_success_message():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(job(), item_name="Filme")

    method, url, kwargs = http.calls[0]
    assert url == "https://ntfy.sh/yuri-topic"
    assert b"Filme" in kwargs["content"]
    assert kwargs["headers"]["Title"] == "Subtitle ready"


def test_failed_job_posts_a_higher_priority_warning():
    http = FakeHTTP()
    Notifier("https://ntfy.sh", "yuri-topic", http=http).job_finished(
        job(state="failed", message="ollama caiu"), item_name="Filme")

    _, _, kwargs = http.calls[0]
    assert kwargs["headers"]["Title"] == "Subtitle job failed"
    assert kwargs["headers"]["Priority"] == "4"
    assert b"ollama caiu" in kwargs["content"]


def test_a_network_error_never_raises():
    Notifier("https://ntfy.sh", "yuri-topic", http=FakeHTTP(raises=ConnectionError())).job_finished(job())
