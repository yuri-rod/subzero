import time

import pytest

from subzero.worker.jobs import MAX_RETRIES, JobStore, Runner


@pytest.fixture
def store(tmp_path):
    return JobStore(str(tmp_path / "jobs.db"))


def test_enqueue_then_pick(store):
    job = store.enqueue("item1", "whisper", "pt-BR")
    assert job.state == "queued"
    assert store.next_queued().id == job.id


def test_manual_jobs_run_before_automatic_ones(store):
    store.enqueue("auto1", "whisper", "pt-BR", origin="auto")
    manual = store.enqueue("manual1", "whisper", "pt-BR")
    assert store.next_queued().id == manual.id


def test_interrupted_jobs_go_back_to_the_queue(store):
    job = store.enqueue("item1", "whisper", "pt-BR")
    store.start(job.id)
    assert store.next_queued() is None

    store.requeue_running()

    assert store.next_queued().id == job.id


def test_cancel_keeps_a_job_from_running(store):
    job = store.enqueue("item1", "whisper", "pt-BR")
    store.cancel(job.id)
    assert store.get(job.id).state == "cancelled"
    assert store.next_queued() is None


def test_progress_is_visible_while_running(store):
    job = store.enqueue("item1", "whisper", "pt-BR")
    store.start(job.id)
    store.progress(job.id, "transcrevendo", 47)

    live = store.get(job.id)
    assert live.state == "running"
    assert live.phase == "transcrevendo"
    assert live.percent == 47


def test_recent_lists_newest_first(store):
    first = store.enqueue("a", "whisper", "pt-BR")
    second = store.enqueue("b", "whisper", "pt-BR")
    assert [j.id for j in store.recent()] == [second.id, first.id]


def test_runner_dispatches_by_kind_and_records_the_result(store):
    job = store.enqueue("item1", "embedded", "pt-BR")
    runner = Runner(store, {"embedded": lambda job, progress: "F:/FILMES/x.pt-BR.srt"})

    done = runner.run_once()

    assert done.id == job.id
    assert store.get(job.id).state == "done"
    assert store.get(job.id).result_path.endswith("x.pt-BR.srt")
    assert store.get(job.id).percent == 100


def test_runner_records_the_failure_message(store):
    job = store.enqueue("item1", "whisper", "pt-BR")

    def blow_up(job, progress):
        raise RuntimeError("sem audio")

    Runner(store, {"whisper": blow_up}).run_once()

    failed = store.get(job.id)
    assert failed.state == "failed"
    assert "sem audio" in failed.message


def test_enqueue_rejects_an_unknown_kind(store):
    with pytest.raises(ValueError):
        store.enqueue("item1", "inventado", "pt-BR")


def test_runner_fails_a_kind_without_a_handler(store):
    job = store.enqueue("item1", "translate", "pt-BR")
    Runner(store, {}).run_once()
    assert store.get(job.id).state == "failed"


def test_runner_with_an_empty_queue_returns_nothing(store):
    assert Runner(store, {}).run_once() is None


def test_handler_progress_reaches_the_store(store):
    job = store.enqueue("item1", "whisper", "pt-BR")
    Runner(store, {"whisper": lambda job, progress: (progress("transcrevendo", 10), "out.srt")[1]}).run_once()
    assert store.get(job.id).result_path == "out.srt"


def test_counts_downloads_of_the_day(store):
    store.enqueue("a", "opensubtitles", "pt-BR")
    job = store.enqueue("b", "opensubtitles", "pt-BR")
    store.start(job.id)
    store.finish(job.id, "b.srt")
    assert store.downloads_today() == 1


def test_manual_job_fails_immediately_even_when_retryable(store):
    job = store.enqueue("item1", "whisper", "pt-BR")
    store.start(job.id)
    store.fail(job.id, "ollama caiu", retryable=True)
    assert store.get(job.id).state == "failed"


def test_auto_job_requeues_with_backoff_instead_of_failing(store):
    job = store.enqueue("item1", "whisper", "pt-BR", origin="auto")
    store.start(job.id)

    store.fail(job.id, "ollama caiu", retryable=True)

    retried = store.get(job.id)
    assert retried.state == "queued"
    assert retried.attempts == 1
    assert retried.next_attempt > time.time()
    assert store.next_queued() is None  # ainda dentro do backoff


def test_auto_job_gives_up_after_max_retries(store):
    job = store.enqueue("item1", "whisper", "pt-BR", origin="auto")
    for _ in range(MAX_RETRIES):
        store.start(job.id)
        store.fail(job.id, "ollama caiu", retryable=True)
        store._set(job.id, next_attempt=0)  # pula o backoff so pra avancar o teste

    store.start(job.id)
    store.fail(job.id, "ollama caiu", retryable=True)

    assert store.get(job.id).state == "failed"


def test_a_handler_missing_kind_never_retries(store):
    job = store.enqueue("item1", "whisper", "pt-BR", origin="auto")
    Runner(store, {}).run_once()
    assert store.get(job.id).state == "failed"


def test_whisper_is_claimed_only_after_everything_else(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.enqueue("a", "whisper", "pt-BR", None, origin="auto")
    store.enqueue("b", "translate", "pt-BR", "en", origin="auto")
    store.enqueue("c", "opensubtitles", "pt-BR", "9", origin="auto")

    order = []
    while (job := store.next_queued()):
        order.append(job.kind)
        store.start(job.id)
        store.finish(job.id, "x")

    assert order[-1] == "whisper"
    assert set(order[:2]) == {"translate", "opensubtitles"}


def test_a_manual_job_still_jumps_the_queue(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.enqueue("a", "translate", "pt-BR", "en", origin="auto")
    store.enqueue("b", "whisper", "pt-BR", None, origin="manual")

    assert store.next_queued().origin == "manual"


def test_next_queued_disallowing_auto(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.enqueue("a", "translate", "pt-BR", "en", origin="auto")
    assert store.next_queued(allow_auto=False) is None
    store.enqueue("b", "whisper", "pt-BR", None, origin="manual")
    assert store.next_queued(allow_auto=False).origin == "manual"
def test_repair_stages_run_cheapest_first(store):
    for kind in ['rebuild','embedded_translate','resync','refetch']:
        store.enqueue(kind,kind,'pt-BR')
    order=[]
    while (job := store.next_queued()):
        order.append(job.kind)
        store.start(job.id)
        store.finish(job.id,'out')
    assert order == ['refetch','resync','embedded_translate','rebuild']


def test_aging_eventually_runs_expensive_jobs(store):
    old = store.enqueue('old','rebuild','pt-BR')
    store._set(old.id,created=time.time()-5*3600)
    store.enqueue('new','refetch','pt-BR')
    assert store.next_queued().id == old.id


def test_two_runners_cannot_claim_parallel_jobs(store):
    store.enqueue('first','refetch','pt-BR')
    store.enqueue('second','refetch','pt-BR')
    assert store.claim() is not None
    assert store.claim() is None


def test_stage_transition_does_not_finish_the_job(store):
    job=store.enqueue('first','refetch','pt-BR')
    def advance(job,progress):
        store.advance(job.id,'resync','No matching download passed')
    Runner(store,{'refetch':advance}).run_once()
    assert store.get(job.id).state == 'queued'
    assert store.get(job.id).kind == 'resync'


def test_cancelled_job_cannot_be_retried_by_exception_handler(store):
    job=store.enqueue('movie','audit','pt-BR',origin='auto')
    store.start(job.id);store.cancel(job.id)
    store.fail(job.id,'cancelled',retryable=True)
    assert store.get(job.id).state=='cancelled'
