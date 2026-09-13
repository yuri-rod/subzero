import pytest

from subzero.worker.applefm import AppleFM, AppleFMGuardrail
from subzero.worker.notify import Notifier
from subzero.worker.triage import cluster, signature, summarize, template_digest


def job(id, state, message, kind="rebuild"):
    return {"id": id, "kind": kind, "state": state, "phase": "", "message": message}


def test_signature_ignores_ids_numbers_and_paths():
    first = signature("Audio rebuild failed: Whisper child exited with status -15: /var/folders/xy/T/subzero-audio-9.wav")
    second = signature("Audio rebuild failed: Whisper child exited with status 3: /var/folders/zz/T/subzero-audio-1.wav")

    assert first == second
    assert "var" not in first and "-15" not in first


def test_signature_keeps_distinct_failures_apart():
    left = signature("Dense caption verification lost a confirmed caption near 3062.593s")
    right = signature("Unstable OCR caption readings near 435.250 seconds")

    assert left != right
    assert "unclosed" not in left


def test_cluster_groups_review_jobs_and_skips_the_rest():
    jobs = [job("a1", "needs_review", "Audio rebuild failed: ollama refused job 7"),
            job("b2", "failed", "Audio rebuild failed: ollama refused job 9"),
            job("c3", "running", "transcrevendo"),
            job("d4", "needs_review", "Unstable OCR caption readings near 5s")]

    groups = cluster(jobs)

    assert [len(g["ids"]) for g in groups] == [2, 1]
    assert groups[0]["kinds"] == ["rebuild"]
    assert "ollama" in groups[0]["example"]


def test_cluster_example_hides_paths_and_ids():
    groups = cluster([job("a1", "failed", "Vision OCR failed for /var/folders/xy/f_009.jpg: boom 12")])

    assert groups[0]["example"] == "Vision OCR failed for : boom"


def test_cluster_accepts_job_objects():
    from types import SimpleNamespace

    jobs = [SimpleNamespace(id="abc123", kind="repair", state="needs_review",
                            phase="", message="Vision OCR failed for f_001.jpg")]

    groups = cluster(jobs)

    assert len(groups) == 1
    assert groups[0]["ids"] == ["abc123"]


def test_template_digest_reports_an_empty_queue():
    assert template_digest([]) == "Review queue is clear."
    assert summarize([], fm=None) == "Review queue is clear."


def test_template_digest_lists_each_group_once():
    groups = cluster([job("a1", "failed", "Audio rebuild failed: ollama refused job 7"),
                      job("b2", "failed", "Audio rebuild failed: ollama refused job 9")])

    digest = template_digest(groups)

    assert digest.startswith("2 jobs need review:")
    assert digest.count("ollama") == 1


class StubFM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.asked = None

    def generate(self, messages, max_tokens=300):
        self.asked = messages
        if self.error is not None:
            raise self.error
        return self.reply


def test_summarize_asks_fm_without_raw_ids():
    groups = cluster([job("abcdef12", "failed", "Audio rebuild failed: ollama refused")])

    digest = summarize(groups, StubFM(reply="Ollama is down."))

    assert digest == "Ollama is down."


def test_summarize_falls_back_to_template_on_refusal():
    groups = cluster([job("a1", "failed", "Audio rebuild failed: ollama refused")])

    digest = summarize(groups, StubFM(error=AppleFMGuardrail("refused")))

    assert digest.startswith("1 jobs need review:")


def test_generate_returns_the_completion_text(monkeypatch):
    fm = AppleFM("http://127.0.0.1:1976")
    sent = {}

    def fake_request(method, path, payload=None):
        sent["payload"] = payload
        return {"choices": [{"message": {"content": "Two jobs failed."}}]}

    monkeypatch.setattr(fm, "_request", fake_request)

    assert fm.generate([{"role": "user", "content": "hi"}], max_tokens=10) == "Two jobs failed."


def test_generate_turns_refusal_into_guardrail(monkeypatch):
    fm = AppleFM("http://127.0.0.1:1976")
    monkeypatch.setattr(fm, "_request",
                        lambda *a, **k: {"choices": [{"message": {"refusal": "nope"}}]})

    with pytest.raises(AppleFMGuardrail):
        fm.generate([{"role": "user", "content": "hi"}])


def test_notifier_digest_posts_with_triage_title():
    sent = {}

    class HTTP:
        def request(self, method, url, **kwargs):
            sent["url"] = url
            sent.update(kwargs)
            return type("R", (), {"status_code": 200})()

    Notifier("https://ntfy.sh", "topic", http=HTTP()).digest("2 jobs need review")

    assert sent["url"] == "https://ntfy.sh/topic"
    assert sent["headers"]["Title"] == "Worker triage"


def test_notifier_digest_stays_quiet_without_topic_or_text():
    class HTTP:
        def request(self, *args, **kwargs):
            raise AssertionError("must not send")

    Notifier("https://ntfy.sh", "", http=HTTP()).digest("text")
    Notifier("https://ntfy.sh", "topic", http=HTTP()).digest("  ")
