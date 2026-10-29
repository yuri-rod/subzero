from dataclasses import replace
from types import SimpleNamespace

import pytest

from subzero.translation_quality import Judgment, JudgeUnavailable
from subzero.worker.jobs import JobStore, Runner
from subzero.worker.service import Service
from subzero.worker.srt import Cue, dump
from subzero.worker.syncflow import SyncFlow
from subzero.worker.translate_cache import read_cached_block, write_cached_block


SOURCES = ["Keep the door closed.", "He never said that.",
           "I left it upstairs.", "She did not come."]
INITIAL = ["Mantenha a porta fechada.", "Ele disse isso.",
           "Deixei lá em cima.", "Ela veio."]
REPAIRED = ["Mantenha a porta fechada.", "Ele nunca disse isso.",
            "Deixei lá em cima.", "Ela não veio."]


class Translator:
    needs_local_compute = False
    provider = "test"
    model = "test"
    supports_context = False

    def __init__(self):
        self.requests = []
        self.output = dict(zip(SOURCES, REPAIRED))
        self.on_request = lambda cues: None
        self.loaded = False

    def ensure_available(self):
        pass

    def release(self):
        self.loaded = False

    def translate_block(self, cues, target_lang, source_lang=None):
        self.loaded = True
        self.requests.append(([c.text for c in cues], target_lang, source_lang))
        self.on_request(cues)
        return [self.output[c.text] for c in cues]


@pytest.fixture
def semantic(tmp_path, monkeypatch):
    translator = Translator()
    service = Service(None, None, None, translator)
    jobs = JobStore(str(tmp_path / "jobs.db"))
    cfg = SimpleNamespace(sync_cache=str(tmp_path / "cache"), semantic_qa=True,
                          ollama_url="http://judge", ollama_model="judge")
    flow = SyncFlow(jobs, service, cfg)

    def judge(pairs, **kwargs):
        return [Judgment(n, 1 if text in (INITIAL[1], INITIAL[3]) else 5,
                         text in (INITIAL[1], INITIAL[3]), "negation changed")
                for n, _, text in pairs]

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", judge)
    source = [Cue(n + 11, n * 3, n * 3 + 2, text) for n, text in enumerate(SOURCES)]
    target = [replace(cue, text=text) for cue, text in zip(source, INITIAL)]
    return flow, translator, source, target


def test_repairs_only_flagged_cues_and_preserves_other_cues(semantic):
    flow, translator, source, target = semantic
    unchanged = target[0], target[2]
    flow.semantic_gate(source, target, "pt-BR", source_lang="en")
    assert [cue.text for cue in target] == REPAIRED
    assert target[0] is unchanged[0] and target[2] is unchanged[1]
    assert [(c.index, c.start, c.end) for c in target] == [
        (11, 0, 2), (12, 3, 5), (13, 6, 8), (14, 9, 11)]
    assert translator.requests == [([SOURCES[1]], "pt-BR", "en"),
                                   ([SOURCES[3]], "pt-BR", "en")]


def test_still_bad_retry_fails_once_without_mutating_output(semantic):
    flow, translator, source, target = semantic
    translator.output = dict(zip(SOURCES, INITIAL))
    with pytest.raises(RuntimeError, match="quality gate"):
        flow.semantic_gate(source, target, "pt-BR")
    assert [cue.text for cue in target] == INITIAL
    assert len(translator.requests) == 2


@pytest.mark.parametrize("accepted", [True, False])
def test_targeted_retry_releases_selected_translator(semantic, accepted):
    flow, translator, source, target = semantic
    if accepted:
        flow.semantic_gate(source, target, "pt-BR")
    else:
        translator.output = dict(zip(SOURCES, INITIAL))
        with pytest.raises(RuntimeError, match="quality gate"):
            flow.semantic_gate(source, target, "pt-BR")
    assert not translator.loaded


def test_retry_must_clear_each_original_defect_even_if_global_score_passes(semantic):
    flow, translator, source, target = semantic
    translator.output[SOURCES[3]] = INITIAL[3]
    with pytest.raises(RuntimeError, match="quality gate"):
        flow.semantic_gate(source, target, "pt-BR")
    assert [cue.text for cue in target] == INITIAL
    assert len(translator.requests) == 2


def test_judge_outage_cannot_accept_previously_rejected_translation(semantic, monkeypatch):
    flow, translator, source, target = semantic

    def after_translation(cues):
        def unavailable(*args, **kwargs):
            raise JudgeUnavailable("offline")
        monkeypatch.setattr("subzero.translation_quality.judge_pairs", unavailable)

    translator.on_request = after_translation
    with pytest.raises(RuntimeError, match="quality gate"):
        flow.semantic_gate(source, target, "pt-BR")
    assert [cue.text for cue in target] == INITIAL
    assert len(translator.requests) == 2


@pytest.mark.parametrize("failure", ["blank", "timing"])
def test_retry_rejects_malformed_output_and_changed_timing(semantic, failure):
    flow, translator, source, target = semantic
    if failure == "blank":
        translator.output[SOURCES[1]] = ""
    else:
        translator.on_request = lambda cues: setattr(cues[0], "start", 999)
    with pytest.raises(RuntimeError):
        flow.semantic_gate(source, target, "pt-BR")
    assert [cue.text for cue in target] == INITIAL
    assert [(c.index, c.start, c.end) for c in source] == [
        (11, 0, 2), (12, 3, 5), (13, 6, 8), (14, 9, 11)]
    assert len(translator.requests) == (2 if failure == "blank" else 1)


def test_cancellation_during_retry_preserves_original_output(semantic):
    flow, translator, source, target = semantic
    job = flow.jobs.enqueue("media", "translate", "pt-BR")
    flow.jobs.start(job.id)
    translator.on_request = lambda cues: flow.jobs.cancel(job.id)
    with pytest.raises(RuntimeError, match="cancel"):
        flow.semantic_gate(source, target, "pt-BR", job=job)
    assert [cue.text for cue in target] == INITIAL
    assert len(translator.requests) == 1


def test_service_translation_propagates_cancellation_into_semantic_retry(semantic):
    flow, translator, source, target = semantic
    flow.service.sync_flow = flow
    job = flow.jobs.enqueue("media", "translate", "pt-BR")
    flow.jobs.start(job.id)
    translator.output = dict(zip(SOURCES, INITIAL))
    translator.on_request = lambda cues: flow.jobs.cancel(job.id) if len(cues) == 1 else None
    with pytest.raises(RuntimeError, match="cancel"):
        flow.service.translate_cues(source, "pt-BR", lambda *_: None, source_lang="en", job=job)
    assert len(translator.requests) == 2


def test_retry_invalidates_only_cached_blocks_containing_flagged_cues(semantic):
    flow, translator, source, target = semantic
    folder = flow.cache / "translations" / "media-key" / "settings"
    folder.mkdir(parents=True)
    for n in range(4):
        write_cached_block(folder, f"{n:06d}", dump(source[n:n + 1]), dump(target[n:n + 1]))
    flow.semantic_gate(source, target, "pt-BR", key="media-key")
    assert read_cached_block(folder, source[0:1], "000000")[0].text == REPAIRED[0]
    assert read_cached_block(folder, source[2:3], "000002")[0].text == REPAIRED[2]
    assert not (folder / "000001.json").exists()
    assert not (folder / "000003.json").exists()
    assert [cue.text for cue in target] == REPAIRED


def test_cancellation_during_successful_judgment_prevents_acceptance(semantic, monkeypatch):
    flow, translator, source, target = semantic
    job = flow.jobs.enqueue("media", "translate", "pt-BR")
    flow.jobs.start(job.id)

    def judge(pairs, **kwargs):
        flow.jobs.cancel(job.id)
        return [Judgment(n, 5, False, "fine") for n, _, _ in pairs]

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", judge)
    with pytest.raises(RuntimeError, match="cancel"):
        flow.semantic_gate(source, target, "pt-BR", job=job)
    assert translator.requests == []


def test_widespread_defects_do_not_retry_but_invalidate_known_bad_cache(semantic):
    flow, translator, _, _ = semantic
    source = [Cue(n + 1, n * 3, n * 3 + 2, "Are you absolutely sure about this decision?")
              for n in range(41)]
    target = [replace(cue, text="Sim.") for cue in source]
    folder = flow.cache / "translations" / "key" / "settings"
    folder.mkdir(parents=True)
    write_cached_block(folder, "000000", dump(source[:20]), dump(target[:20]))
    with pytest.raises(RuntimeError, match="quality gate"):
        flow.semantic_gate(source, target, "pt-BR", key="key")
    assert translator.requests == []
    assert not (folder / "000000.json").exists()


def test_judge_resource_pressure_defers_job_instead_of_accepting_fallback(semantic, monkeypatch):
    from contextlib import contextmanager
    from subzero.compute import ComputePressureError

    flow, translator, source, target = semantic
    job = flow.jobs.enqueue("media", "translate", "pt-BR")

    @contextmanager
    def pressure(*args, **kwargs):
        raise ComputePressureError("Memory pressure", retry_after=45)
        yield

    monkeypatch.setattr("subzero.worker.syncflow.compute_phase", pressure)
    runner = Runner(flow.jobs, {"translate": lambda job, progress:
                    flow.semantic_gate(source, target, "pt-BR", job=job)})
    result = runner.run_once()
    assert result.state == "queued"
    assert result.phase == "resource_pressure"
    assert result.attempts == 0
    assert result.next_attempt > result.updated
    assert [cue.text for cue in target] == INITIAL


def test_targeted_retry_resource_pressure_reaches_runner_without_review(semantic, tmp_path):
    from subzero.compute import ComputePressureError

    flow, translator, source, target = semantic
    job = flow.jobs.enqueue("media", "translate", "pt-BR")
    media = SimpleNamespace(path=str(tmp_path / "movie.mkv"), item_id="media")
    translator.output = dict(zip(SOURCES, INITIAL))

    def pressure_on_singleton(cues):
        if len(cues) == 1:
            raise ComputePressureError("Memory pressure", retry_after=45)

    translator.on_request = pressure_on_singleton
    runner = Runner(flow.jobs, {"translate": lambda job, progress:
                    flow.translate_source(media, job, "key", {}, progress, dump(source), "en")})
    result = runner.run_once()
    assert result.state == "queued"
    assert result.phase == "resource_pressure"
    assert result.attempts == 0
    assert result.next_attempt > result.updated
    assert not list(tmp_path.glob("*.srt"))
