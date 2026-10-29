import json

import pytest

from subzero.translation_quality import (
    SAMPLE_CUES,
    JudgeUnavailable,
    Verdict,
    check_translation,
    deterministic_hits,
    deterministic_verdict,
    judge_pairs,
    parse_judgments,
    sample_indexes,
)


def test_flags_extreme_length_ratio():
    hits = deterministic_hits(
        ["Are you absolutely sure about this decision right now?"],
        ["Sim."],
    )
    assert 0 in hits


def test_ignores_normal_length_ratio():
    hits = deterministic_hits(
        ["Are you sure about this?"],
        ["Você tem certeza disso?"],
    )
    assert hits == {}


@pytest.mark.parametrize("source, translated", [
    ("絶対に許さない。", "Eu nunca vou perdoar você por isso."),
    ("我绝不会原谅你。", "Eu nunca vou perdoar você por isso."),
    ("絕不原諒你。", "Eu nunca vou perdoar você por isso."),
    ("절대 용서 안 해.", "Eu nunca vou perdoar você por isso."),
    ("ヤメテ。", "Por favor, pare de fazer isso agora!"),
    ("ﾔﾒﾃ｡", "Por favor, pare de fazer isso agora!"),
    ("NASAに行きたい。", "Eu quero ir para a agência espacial NASA."),
    ("<i>絶対に許さない。</i>", "<i>Eu nunca vou perdoar você por isso.</i>"),
])
def test_ignores_cjk_to_portuguese_expansion(source, translated):
    assert deterministic_hits([source], [translated]) == {}


def test_cjk_expansion_passes_gate_with_good_judge_or_outage(monkeypatch):
    sources = ["絶対に許さない。"] * 5
    translated = ["Eu nunca vou perdoar você por isso."] * 5
    assert deterministic_verdict(sources, translated).ok
    monkeypatch.setattr("subzero.translation_quality.judge_pairs", good_judge)
    assert check_translation(sources, translated, url="http://127.0.0.1:11434", model="m").ok

    def down(_pairs, **_kwargs):
        raise JudgeUnavailable("ollama is down")

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", down)
    verdict = check_translation(sources, translated, url="http://127.0.0.1:11434", model="m")
    assert verdict.ok
    assert verdict.score is None


@pytest.mark.parametrize("source, translated, reason", [
    ("明日の朝までにこの仕事を全部終わらせなければならない。", "Sim.", "extreme length ratio"),
    ("絶対に許さない。", "", "empty translation"),
    ("Hi, 李.", "Oi, Li. Que bom encontrar você por aqui hoje!", "extreme length ratio"),
    ("Hi!", "Olá! Que bom encontrar você por aqui hoje!", "extreme length ratio"),
    ("行け。", "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ", "extreme length ratio"),
])
def test_script_exception_preserves_other_length_and_empty_checks(source, translated, reason):
    assert reason in deterministic_hits([source], [translated])[0]


def test_ignores_short_cues_and_flags_empty_translation():
    assert deterministic_hits(["Yes!"], ["Sim, é isso mesmo!"]) == {}
    hits = deterministic_hits(["Where are we going tonight?"], [""])
    assert 0 in hits


def test_flags_mistranslated_game_terms():
    hits = deterministic_hits(
        ["He found an immunity idol on the beach."],
        ["Ele encontrou um prêmio na praia."],
    )
    assert 0 in hits
    hits = deterministic_hits(
        ["See you at tribal council tonight."],
        ["Vejo você na reunião hoje à noite."],
    )
    assert 0 in hits


def test_accepts_correct_game_terms():
    hits = deterministic_hits(
        ["He found an immunity idol.", "See you at tribal council."],
        ["Ele encontrou um ídolo de imunidade.", "Vejo você no conselho tribal."],
    )
    assert hits == {}


def test_flags_repeated_translation_for_distinct_dialogue():
    sources = [
        "The weather is beautiful this morning.",
        "I need to buy some groceries today.",
        "The car broke down on the highway.",
    ]
    hits = deterministic_hits(sources, ["Está tudo bem por aqui."] * 3)
    assert set(hits) == {0, 1, 2}


def test_ignores_repeated_short_acknowledgments():
    sources = ["Yes.", "Yeah.", "Yes, I agree."]
    hits = deterministic_hits(sources, ["Sim."] * 3)
    assert hits == {}


def test_sample_spans_file_and_keeps_suspects():
    sample = sample_indexes(100, [3, 77])
    assert len(sample) == SAMPLE_CUES
    assert 3 in sample and 77 in sample
    assert sample[0] < 10
    assert sample[-1] > 89
    assert sample == sorted(sample)
    assert sample_indexes(100, [3, 77]) == sample


def test_sample_caps_at_forty_with_many_suspects():
    sample = sample_indexes(100, list(range(60)))
    assert len(sample) == SAMPLE_CUES


def test_sample_returns_everything_when_short():
    assert sample_indexes(12, [4]) == list(range(12))


def good_judge(pairs, **kwargs):
    from subzero.translation_quality import Judgment
    return [Judgment(index, 5, False, "same meaning") for index, _, _ in pairs]


def bad_judge(pairs, **kwargs):
    from subzero.translation_quality import Judgment
    return [Judgment(index, 2, False, "wrong meaning") for index, _, _ in pairs]


def test_clean_translation_passes(monkeypatch):
    monkeypatch.setattr("subzero.translation_quality.judge_pairs", good_judge)
    verdict = check_translation(
        ["Are you sure about this?"], ["Você tem certeza disso?"],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert verdict.ok
    assert verdict.score == 5.0


def test_low_mean_score_needs_review(monkeypatch):
    monkeypatch.setattr("subzero.translation_quality.judge_pairs", bad_judge)
    verdict = check_translation(
        ["Are you sure?", "Where is he?"], ["Você tem certeza?", "Onde ele está?"],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert not verdict.ok
    assert verdict.score == 2.0
    assert "2.0" in verdict.reason


def test_verdict_identifies_all_low_scoring_and_critical_cues(monkeypatch):
    from subzero.translation_quality import Judgment

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", lambda pairs, **kwargs: [
        Judgment(0, 5, False, "fine"), Judgment(1, 3, False, "lost meaning"),
        Judgment(2, 5, True, "wrong person"), Judgment(3, 5, True, "negated"),
    ])
    verdict = check_translation(["Hello there."] * 4, ["Olá, tudo bem?"] * 4,
                                url="http://judge", model="m")
    assert not verdict.ok
    assert verdict.defect_indexes == (1, 2, 3)


def test_deterministic_verdict_identifies_defects_during_judge_outage(monkeypatch):
    def unavailable(*args, **kwargs):
        raise JudgeUnavailable("offline")

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", unavailable)
    verdict = check_translation(["Are you absolutely sure about this decision?"] * 5,
                                ["Sim."] * 5, url="http://judge", model="m")
    assert not verdict.ok
    assert verdict.defect_indexes == (0, 1, 2, 3, 4)


def test_recheck_keeps_previously_defective_cues_in_sample(monkeypatch):
    from subzero.translation_quality import Judgment

    def judge(pairs, **kwargs):
        return [Judgment(n, 1 if n in (77, 79) else 5, n in (77, 79), "meaning")
                for n, _, _ in pairs]

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", judge)
    verdict = check_translation(["Hello there."] * 100, ["Olá, tudo bem?"] * 100,
                                url="http://judge", model="m", required_indexes=(77, 79))
    assert not verdict.ok
    assert verdict.defect_indexes == (77, 79)


def test_two_criticals_need_review_even_with_high_mean(monkeypatch):
    from subzero.translation_quality import Judgment

    def critical_judge(pairs, **kwargs):
        out = []
        for n, (index, _source, _translated) in enumerate(pairs):
            out.append(Judgment(index, 5, n < 2, "critical" if n < 2 else "fine"))
        return out

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", critical_judge)
    sources = [f"Line number {n} here." for n in range(10)]
    verdict = check_translation(
        sources, [f"Linha número {n} aqui." for n in range(10)],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert not verdict.ok
    assert len(verdict.worst) <= 5


def test_judge_outage_degrades_to_deterministic_pass(monkeypatch):
    def down(_pairs, **_kwargs):
        raise JudgeUnavailable("ollama is down")

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", down)
    verdict = check_translation(
        ["Are you sure about this?"], ["Você tem certeza disso?"],
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert isinstance(verdict, Verdict)
    assert verdict.ok
    assert verdict.score is None


def test_judge_outage_with_many_suspects_needs_review(monkeypatch):
    def down(_pairs, **_kwargs):
        raise JudgeUnavailable("ollama is down")

    monkeypatch.setattr("subzero.translation_quality.judge_pairs", down)
    sources = ["Are you absolutely sure about this decision right now?"] * 6
    verdict = check_translation(
        sources, ["Sim."] * 6,
        url="http://127.0.0.1:11434", model="qwen3.5:9b-mlx",
    )
    assert not verdict.ok
    assert verdict.score is None


def test_mismatched_cue_counts_raise():
    with pytest.raises(ValueError):
        check_translation(["one", "two"], ["um"],
                            url="http://127.0.0.1:11434", model="m")


def test_parse_judgments_accepts_valid_batch():
    body = {"response": json.dumps([
        {"id": 4, "adequacy": 5, "critical": False, "reason": "same meaning"},
        {"id": 9, "adequacy": 2, "critical": True, "reason": "negation dropped"},
    ])}
    judgments = parse_judgments([4, 9], body)
    assert [(j.cue, j.adequacy, j.critical) for j in judgments] == [(4, 5, False), (9, 2, True)]


@pytest.mark.parametrize("body", [
    {"response": "not json"},
    {"response": json.dumps([{"id": 4, "adequacy": 9, "critical": False, "reason": "x"}])},
    {"response": json.dumps([{"id": 4, "adequacy": 5, "critical": False, "reason": "x"}])},
    {"nope": True},
])
def test_parse_judgments_rejects_garbage(body):
    with pytest.raises(JudgeUnavailable):
        parse_judgments([4, 9], body)


def test_flags_second_game_term_when_first_is_correct():
    hits = deterministic_hits(
        ["He hid the immunity idol before tribal council."],
        ["Ele escondeu o ídolo de imunidade antes da reunião."],
    )
    assert 0 in hits
    assert any("tribal council" in reason for reason in hits[0])


def test_truncated_judge_response_degrades_to_deterministic(monkeypatch):
    import http.client

    def truncated(_request, timeout=None):
        raise http.client.IncompleteRead(partial=b'{"resp', expected=100)

    monkeypatch.setattr("subzero.translation_quality.urllib.request.urlopen", truncated)
    verdict = check_translation(
        ["Are you sure about this?"], ["Você tem certeza disso?"],
        url="http://127.0.0.1:11434", model="m",
    )
    assert verdict.ok
    assert verdict.score is None


def test_deterministic_rejection_names_suspect_causes(monkeypatch):
    monkeypatch.setattr("subzero.translation_quality.judge_pairs", good_judge)
    sources = ["Are you absolutely sure about this decision right now?"] * 6
    verdict = check_translation(
        sources, ["Sim."] * 6,
        url="http://127.0.0.1:11434", model="m",
    )
    assert not verdict.ok
    assert verdict.score == 5.0
    assert "deterministic" in verdict.reason
    assert "length ratio" in verdict.reason


def test_deterministic_verdict_passes_clean_translation():
    verdict = deterministic_verdict(
        ["Are you sure about this?"], ["Você tem certeza disso?"]
    )
    assert verdict.ok
    assert verdict.score is None


def test_deterministic_verdict_rejects_many_suspects():
    sources = ["Are you absolutely sure about this decision right now?"] * 6
    verdict = deterministic_verdict(sources, ["Sim."] * 6)
    assert not verdict.ok
    assert "6 deterministic suspect cues" in verdict.reason
    assert "length ratio" in verdict.reason


def test_judge_pairs_batches_twenty_per_request(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, _size=-1):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        ids = [item["id"] for item in json.loads(payload["prompt"].rsplit("PAIRS:", 1)[1])]
        calls.append(len(ids))
        return FakeResponse({"response": json.dumps([
            {"id": i, "adequacy": 4, "critical": False, "reason": "ok"} for i in ids
        ])})

    monkeypatch.setattr("subzero.translation_quality.urllib.request.urlopen", fake_urlopen)
    pairs = [(n, f"source {n}", f"tradução {n}") for n in range(40)]
    judgments = judge_pairs(pairs, url="http://x", model="m")
    assert calls == [20, 20]
    assert len(judgments) == 40


def test_openai_judge_transport_returns_localized_defects(monkeypatch):
    import io

    def respond(request, timeout=None):
        assert request.full_url == "http://localhost:8080/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-key"
        payload = json.loads(request.data)
        assert payload["model"] == "local-judge"
        response_format = payload["response_format"]
        assert response_format["type"] == "json_schema"
        schema = response_format["json_schema"]["schema"]
        assert schema["required"] == ["judgments"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["judgments"]["minItems"] == 1
        assert schema["properties"]["judgments"]["maxItems"] == 1
        assert '"judgments"' in payload["messages"][0]["content"]
        pairs = json.loads(payload["messages"][0]["content"].rsplit("PAIRS:", 1)[1])
        assert pairs == [{"id": 0, "en": "He never came.", "pt": "Ele veio."}]
        content = json.dumps({"judgments": [
            {"id": 0, "adequacy": 1, "critical": True, "reason": "negation"}]})
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop",
                                                  "message": {"content": content}}]}).encode())

    monkeypatch.setattr("subzero.translation_quality.urllib.request.urlopen", respond)
    verdict = check_translation(["He never came."], ["Ele veio."], provider="openai",
                                url="http://localhost:8080/v1/", model="local-judge", api_key="test-key")
    assert not verdict.ok
    assert verdict.score == 1
    assert verdict.defect_indexes == (0,)


@pytest.mark.parametrize("response", [
    {"choices": []},
    {"choices": [{"finish_reason": "length", "message": {"content": "[]"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": '{"judgments": [], "extra": true}'}}]},
])
def test_invalid_openai_judge_response_degrades_to_deterministic(monkeypatch, response):
    import io

    monkeypatch.setattr("subzero.translation_quality.urllib.request.urlopen",
                        lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode()))
    verdict = check_translation(["Hello there."], ["Olá, tudo bem?"], provider="openai",
                                url="http://localhost:8080/v1", model="local-judge")
    assert verdict.ok
    assert verdict.score is None
    assert "judge unavailable" in verdict.reason


def test_judge_rejects_duplicate_ids_instead_of_overwriting_a_defect():
    body = {"response": json.dumps([
        {"id": 0, "adequacy": 1, "critical": True, "reason": "wrong meaning"},
        {"id": 0, "adequacy": 5, "critical": False, "reason": "fine"},
    ])}
    with pytest.raises(JudgeUnavailable):
        parse_judgments([0], body)
