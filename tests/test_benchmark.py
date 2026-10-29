import json
import http.client
from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest


def test_benchmark_aggregates_real_outputs_excludes_warmup_and_keeps_unknown_tokens(monkeypatch):
    from subzero import benchmark
    responses = iter([['Olá.'], ['Olá.'], ['Olá.']])

    class Client:
        model = 'fixture'
        usage = {'completion_tokens': None, 'prompt_tokens': None}

        def translate_block(self, cues, target_lang, **kwargs):
            return next(responses)

    monkeypatch.setattr(benchmark, '_client', lambda variant: Client())
    monkeypatch.setattr(benchmark, 'sample_resources', lambda: {
        'memory_total_bytes': 100, 'memory_available_bytes': 20, 'swap_used_bytes': 7,
        'vram_free_bytes': None})
    ticks = iter([0, 2, 2, 6])
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: next(ticks))
    report = benchmark.run_benchmark({'variants': [{'provider': 'openai', 'model': 'fixture'}],
                                      'warmup': 1, 'repeats': 2}, sources=['Hello.'])
    result = report['variants'][0]
    assert [run['elapsed_seconds'] for run in result['runs']] == [2, 4]
    assert result['metrics']['latency_p50_seconds'] == 3
    assert result['metrics']['latency_p95_seconds'] == pytest.approx(3.9)
    assert result['metrics']['cues_per_second'] == pytest.approx(1 / 3)
    assert result['metrics']['tokens_per_second'] is None
    assert result['metrics']['guardrail_pass_rate'] == 1
    assert result['resources']['peak_memory_used_bytes'] == 80
    assert result['resources']['peak_swap_used_bytes'] == 7
    assert result['metrics']['judge_mean_adequacy'] is None


def test_benchmark_records_failed_run_and_judge_outage_without_fabricated_scores(monkeypatch):
    from subzero import benchmark

    class Client:
        model = 'fixture'
        usage = {'completion_tokens': 12, 'prompt_tokens': 20}

        def translate_block(self, cues, target_lang, **kwargs):
            return ['Olá.']

    monkeypatch.setattr(benchmark, '_client', lambda variant: Client())
    monkeypatch.setattr(benchmark, 'sample_resources', lambda: {})
    monkeypatch.setattr(benchmark, 'check_translation', lambda *a, **k: SimpleNamespace(
        ok=True, score=None, reason='judge unavailable', defect_indexes=()))
    report = benchmark.run_benchmark({'variants': [{'provider': 'openai', 'model': 'fixture'}],
                                      'warmup': 0, 'repeats': 1,
                                      'judge': {'model': 'judge', 'url': 'http://localhost:9999'}},
                                     sources=['Hello.'])
    assert report['variants'][0]['metrics']['judge_mean_adequacy'] is None
    assert report['variants'][0]['runs'][0]['judge']['score'] is None
    assert 'unavailable' in report['variants'][0]['runs'][0]['judge']['reason']
    monkeypatch.setattr(Client, 'translate_block', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')))
    failed = benchmark.run_benchmark({'variants': [{'provider': 'openai', 'model': 'fixture'}],
                                      'warmup': 0, 'repeats': 1}, sources=['Hello.'])
    assert failed['variants'][0]['metrics']['successful_runs'] == 0
    assert failed['variants'][0]['metrics']['guardrail_pass_rate'] == 0
    assert failed['variants'][0]['runs'][0]['error'] == 'RuntimeError: offline'


def test_benchmark_cli_writes_json_and_markdown_without_secrets(monkeypatch, tmp_path):
    from subzero import benchmark
    from subzero.cli import main

    class Client:
        model = 'fixture'
        usage = {'completion_tokens': 3, 'prompt_tokens': 4}

        def translate_block(self, cues, target_lang, **kwargs):
            return ['Olá.'] * len(cues)

    monkeypatch.setattr(benchmark, '_client', lambda variant: Client())
    monkeypatch.setattr(benchmark, 'sample_resources', lambda: {})
    monkeypatch.setenv('BENCH_KEY', 'a-secret-value')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'variants': [{'provider': 'openai', 'model': 'fixture',
                                               'api_key_env': 'BENCH_KEY'}], 'repeats': 1, 'warmup': 0}))
    output = tmp_path / 'report.json'
    assert main(['benchmark', '--config', str(config), '--output', str(output)]) == 0
    data = json.loads(output.read_text())
    assert data['workload']['sha256']
    assert len(data['variants'][0]['runs']) == 1
    assert 'a-secret-value' not in output.read_text()
    markdown = output.with_suffix('.md').read_text()
    assert 'p95' in markdown and 'fixture' in markdown and 'Peak swap' in markdown


@pytest.mark.parametrize('patch', [{'repeats': 0}, {'sample_interval': 0}, {'warmup': -1},
                                  {'repeats': True}, {'sample_interval': float('nan')},
                                  {'variants': []}, {'judge': []}])
def test_benchmark_rejects_invalid_configuration(patch):
    from subzero import benchmark
    config = {'variants': [{'provider': 'openai', 'model': 'fixture'}], **patch}
    with pytest.raises(ValueError):
        benchmark.run_benchmark(config)


def test_benchmark_rejects_credentials_in_report_endpoint():
    from subzero import benchmark
    with pytest.raises(ValueError):
        benchmark.run_benchmark({'variants': [{'provider': 'openai', 'model': 'fixture',
                                              'url': 'https://secret@example.com/v1'}]})


@pytest.mark.parametrize('provider', ['openai', 'ollama'])
def test_benchmark_real_adapters_capture_usage_and_lease_judge(monkeypatch, provider):
    from subzero import benchmark, translate
    active = []
    calls = []

    @contextmanager
    def phase(kind, **kwargs):
        active.append(kind)
        try:
            yield
        finally:
            active.pop()

    def request(req, **kwargs):
        payload = json.loads(req.data)
        calls.append(payload['model'])
        if payload['model'] == 'judge':
            assert active == ['ollama']
            body = {'response': json.dumps([{'id': i, 'adequacy': 5, 'critical': False,
                                             'reason': 'preserved'} for i in range(2)])}
        elif provider == 'openai':
            assert not active
            body = {'choices': [{'message': {'content': '1. Olá.'}, 'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 4, 'prompt_tokens': 10}}
        else:
            assert active
            body = {'response': '1. Olá.', 'done': True, 'done_reason': 'stop',
                    'eval_count': 4, 'prompt_eval_count': 10}
        response = BytesIO(json.dumps(body).encode())
        response.status = 200
        return response

    monkeypatch.setattr(benchmark, 'compute_phase', phase)
    monkeypatch.setattr(translate, 'compute_phase', phase)
    monkeypatch.setattr('urllib.request.urlopen', request)
    monkeypatch.setattr(benchmark, 'sample_resources', lambda: {})
    report = benchmark.run_benchmark({'variants': [{'provider': provider, 'model': 'fixture'}],
                                     'warmup': 0, 'repeats': 1, 'batch_size': 1,
                                     'judge': {'model': 'judge'}}, sources=['Hello.', 'Hi.'])
    run = report['variants'][0]['runs'][0]
    assert calls == ['fixture', 'fixture', 'judge']
    assert run['completion_tokens'] == 8 and run['prompt_tokens'] == 20
    assert run['judge']['score'] == 5
    assert not active


def test_benchmark_saves_interrupted_http_runs(monkeypatch, tmp_path):
    from subzero import benchmark
    from subzero.cli import main

    class TruncatedResponse(BytesIO):
        status = 200

        def read(self, *args):
            raise http.client.IncompleteRead(b'partial response')

    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: TruncatedResponse())
    monkeypatch.setattr(benchmark, 'sample_resources', lambda: {})
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'variants': [{'provider': 'openai', 'model': 'fixture'}],
                                  'warmup': 1, 'repeats': 1}))
    output = tmp_path / 'report.json'
    assert main(['benchmark', '--config', str(config), '--output', str(output)]) == 1
    result = json.loads(output.read_text())['variants'][0]
    assert result['warmup_errors'][0].startswith('IncompleteRead:')
    assert result['runs'][0]['error'].startswith('IncompleteRead:')
    assert result['metrics']['successful_runs'] == 0
    assert result['metrics']['guardrail_pass_rate'] == 0
