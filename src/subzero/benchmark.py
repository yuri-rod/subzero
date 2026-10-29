"""Repeatable subtitle workload measurements against explicitly selected servers."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import platform
import tempfile
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .compute import check_compute_pressure, compute_phase, sample_resources
from .convert import Cue, dump_srt
from .translate import TRANSLATION_PROMPT_VERSION, OllamaClient, OpenAIClient, translate_cues
from .translation_quality import check_translation, deterministic_hits
from .worker.guards import check_excellence_guards
from .worker.srt import timestamp

WORKLOAD_VERSION = 'subtitle-dialogue-v1'
SOURCES = (
    'Please close the door.', 'I did not take your keys.',
    'We have three minutes left.', 'Ana will meet us tomorrow.',
    'You promised to tell the truth.', 'Keep an eye on the children.',
    'The train leaves at six.', 'I thought you were joking.',
    'Nobody knows where he went.', 'Could you speak more slowly?',
    'This is our last chance.', 'Thank you for coming back.',
)


def _endpoint(url):
    if not isinstance(url, str):
        raise ValueError('Benchmark endpoints must be strings')
    parsed = urlsplit(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError('Benchmark endpoints must be HTTP(S) URLs without credentials, query or fragment')
    return url.rstrip('/')


def _variant(raw):
    if not isinstance(raw, dict) or raw.get('provider') not in ('ollama', 'openai'):
        raise ValueError('Each variant requires provider ollama or openai')
    if not isinstance(raw.get('model'), str) or not raw['model'].strip():
        raise ValueError('Each variant requires a model name')
    provider = raw['provider']
    if any(not isinstance(raw[key], str) for key in (
            'label', 'precision', 'api_key_env', 'runtime', 'model_file') if key in raw):
        raise ValueError('Variant identity and credential variable names must be strings')
    url = _endpoint(raw.get('url', 'http://127.0.0.1:11434' if provider == 'ollama'
                            else 'http://127.0.0.1:8080/v1'))
    out = {name: raw[name] for name in ('model', 'provider', 'label', 'precision', 'api_key_env',
                                       'runtime', 'model_file') if name in raw}
    out.update(url=url, label=raw.get('label', raw['model']), precision=raw.get('precision', 'unspecified'))
    return out


def _client(variant):
    if variant['provider'] == 'ollama':
        return OllamaClient(url=variant['url'], model=variant['model'])
    return OpenAIClient(base_url=variant['url'], model=variant['model'],
                        api_key=os.environ.get(variant.get('api_key_env', 'OPENAI_API_KEY'), ''))


class ResourceSamples:
    """Sample host pressure during translation, excluding judge execution."""

    def __init__(self, interval):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()

    def sample(self):
        try:
            self.samples.append(sample_resources())
        except (OSError, ValueError, RuntimeError) as err:
            self.samples.append({'error': type(err).__name__})

    def __enter__(self):
        self.sample()

        def monitor():
            while not self.stop.wait(self.interval):
                self.sample()

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()


def _percentile(values, percentile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _resources(samples):
    def extreme(name, fn=max):
        values = [s[name] for s in samples if s.get(name) is not None]
        return fn(values) if values else None

    used = [s['memory_total_bytes'] - s['memory_available_bytes'] for s in samples
            if s.get('memory_total_bytes') is not None and s.get('memory_available_bytes') is not None]
    return {'sample_count': len(samples), 'peak_memory_used_bytes': max(used) if used else None,
            'min_memory_available_bytes': extreme('memory_available_bytes', min),
            'peak_swap_used_bytes': extreme('swap_used_bytes'),
            'min_vram_free_bytes': extreme('vram_free_bytes', min)}


def run_benchmark(config, *, sources=None):
    """Measure complete workload repetitions; retain failed runs and missing metrics."""
    if not isinstance(config, dict):
        raise ValueError('Benchmark configuration must be an object')
    repeats, warmup, batch_size = (config.get('repeats', 3), config.get('warmup', 1),
                                  config.get('batch_size', 4))
    for value, minimum, name in ((repeats, 1, 'repeats'), (warmup, 0, 'warmup'), (batch_size, 1, 'batch_size')):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    interval = config.get('sample_interval', 0.25)
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('sample_interval must be positive and finite')
    raw_variants = config.get('variants')
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError('At least one variant is required')
    variants = [_variant(v) for v in raw_variants]
    judge = config.get('judge')
    if judge is not None:
        if not isinstance(judge, dict):
            raise ValueError('judge must be an object')
        judge = _variant({'provider': 'ollama', **judge})
    sources = list(SOURCES if sources is None else sources)
    if not sources or any(not isinstance(s, str) or not s.strip() for s in sources):
        raise ValueError('The workload must contain nonempty cue strings')
    target_lang = config.get('target_lang', 'pt-BR')
    if judge and target_lang not in ('pt', 'pt-BR', 'pt-br', 'por', 'pob'):
        raise ValueError('The adequacy judge supports Portuguese targets')
    source_lang = config.get('source_lang', 'en')
    cues = [Cue(timestamp(i * 3), timestamp(i * 3 + 2), text) for i, text in enumerate(sources)]
    report = {
        'schema_version': 1, 'started_at': datetime.now(timezone.utc).isoformat(),
        'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                     'cpu_count': os.cpu_count(), 'python': platform.python_version()},
        'workload': {'version': WORKLOAD_VERSION if sources == list(SOURCES) else 'custom',
                     'sha256': hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest(),
                     'sources': sources, 'cue_count': len(cues)},
        'settings': {'repeats': repeats, 'warmup': warmup, 'batch_size': batch_size,
                     'target_lang': target_lang, 'source_lang': source_lang,
                     'sample_interval': interval, 'prompt_version': TRANSLATION_PROMPT_VERSION,
                     'judge': judge}, 'variants': [],
    }
    for variant in variants:
        identity = {k: v for k, v in variant.items() if k not in ('api_key_env', 'model_file')}
        if variant.get('model_file'):
            digest = hashlib.sha256()
            with Path(variant['model_file']).open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            identity['model_sha256'] = digest.hexdigest()
        warmup_errors = []
        for _ in range(warmup):
            check_compute_pressure()
            try:
                translate_cues(cues, target_lang, _client(variant), batch_size=batch_size, source_lang=source_lang)
            except (OSError, ValueError, RuntimeError, http.client.HTTPException) as err:
                warmup_errors.append(f'{type(err).__name__}: {err}')
        runs, samples = [], []
        for _ in range(repeats):
            check_compute_pressure()
            client = _client(variant)
            translated, error = [], None
            with ResourceSamples(interval) as sampler:
                start = time.perf_counter()
                try:
                    translated = translate_cues(cues, target_lang, client, batch_size=batch_size, source_lang=source_lang)
                except (OSError, ValueError, RuntimeError, http.client.HTTPException) as err:
                    error = f'{type(err).__name__}: {err}'
                elapsed = time.perf_counter() - start
            samples.extend(sampler.samples)
            texts = [c.text for c in translated]
            det = deterministic_hits(sources, texts) if len(texts) == len(sources) else {}
            passed = sum(i not in det and check_excellence_guards(dump_srt([cue]), target_lang).ok
                         for i, cue in enumerate(translated))
            verdict = None
            if judge and not error:
                check_compute_pressure()
                phase = (compute_phase('ollama', ollama_url=judge['url'])
                         if judge['provider'] == 'ollama' else nullcontext())
                with phase:
                    verdict = check_translation(sources, texts, url=judge['url'], model=judge['model'],
                                                provider=judge['provider'],
                                                api_key=os.environ.get(judge.get('api_key_env', 'OPENAI_API_KEY'), ''))
            usage = getattr(client, 'usage', {})
            runs.append({'elapsed_seconds': elapsed, 'error': error, 'output': texts,
                         'completion_tokens': usage.get('completion_tokens'),
                         'prompt_tokens': usage.get('prompt_tokens'), 'guardrail_passes': passed,
                         'resources': sampler.samples,
                         'judge': None if verdict is None else {'ok': verdict.ok, 'score': verdict.score,
                                                               'reason': verdict.reason}})
        elapsed = sum(r['elapsed_seconds'] for r in runs)
        tokens = [r['completion_tokens'] for r in runs]
        scores = [r['judge']['score'] for r in runs if r['judge'] and r['judge']['score'] is not None]
        successful = sum(r['error'] is None for r in runs)
        metrics = {'successful_runs': successful, 'attempted_runs': repeats,
                   'latency_p50_seconds': _percentile([r['elapsed_seconds'] for r in runs], .5),
                   'latency_p95_seconds': _percentile([r['elapsed_seconds'] for r in runs], .95),
                   'cues_per_second': successful * len(cues) / elapsed if elapsed else None,
                   'tokens_per_second': sum(tokens) / elapsed if elapsed and all(t is not None for t in tokens) else None,
                   'guardrail_pass_rate': sum(r['guardrail_passes'] for r in runs) / (repeats * len(cues)),
                   'judge_scored_runs': len(scores),
                   'judge_mean_adequacy': sum(scores) / len(scores) if scores else None}
        report['variants'].append({**identity, 'warmup_errors': warmup_errors, 'metrics': metrics,
                                   'resources': _resources(samples), 'runs': runs})
    return report


def markdown_report(report):
    def fmt(value):
        return 'unavailable' if value is None else f'{value:.3f}'

    lines = ['# Subtitle inference benchmark', '', f"Workload: `{report['workload']['version']}`",
             f"Workload SHA-256: `{report['workload']['sha256']}`", '',
             '| Variant | Precision | Runs passed | p50 (s) | p95 (s) | Cues/s | Tokens/s | Guardrail rate | Adequacy /5 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for variant in report['variants']:
        m = variant['metrics']
        label = str(variant['label']).replace('|', '\\|').replace('\n', ' ')
        precision = str(variant['precision']).replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {label} | {precision} | {m['successful_runs']}/{m['attempted_runs']} | "
                     + ' | '.join(fmt(m[key]) for key in ('latency_p50_seconds', 'latency_p95_seconds',
                        'cues_per_second', 'tokens_per_second', 'guardrail_pass_rate', 'judge_mean_adequacy')) + ' |')
    lines += ['', '| Variant | Peak host memory used (GiB) | Peak swap used (GiB) | Minimum free VRAM (GiB) |',
              '| --- | --- | --- | --- |']
    for variant in report['variants']:
        values = [variant['resources'][key] for key in
                  ('peak_memory_used_bytes', 'peak_swap_used_bytes', 'min_vram_free_bytes')]
        label = str(variant['label']).replace('|', '\\|').replace('\n', ' ')
        lines.append(f'| {label} | ' + ' | '.join(
            fmt(value / 1024 ** 3 if value is not None else None) for value in values) + ' |')
    lines += ['', 'Latency covers each complete translation workload, including client retries. Warmup and judge time are excluded.',
              'Token throughput uses provider-reported completion tokens divided by translation wall time; missing counts stay unavailable.',
              'Memory and swap are sampled host-wide during translation, not isolated model allocations. Inspect JSON for samples, failures, outputs and scored-run counts.', '']
    return '\n'.join(lines)


def write_report(report, output):
    output = Path(output)
    if output.suffix != '.json':
        raise ValueError('Benchmark output must end in .json; Markdown is written beside it')
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, text in ((output, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       (output.with_suffix('.md'), markdown_report(report))):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            try:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                os.replace(tmp, path)
            finally:
                stream.close()
                tmp.unlink(missing_ok=True)


def command(args):
    try:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        report = run_benchmark(config)
        write_report(report, args.output)
    except (OSError, ValueError, RuntimeError, http.client.HTTPException) as err:
        print(f'subzero benchmark: {err}')
        return 1
    print(f'Benchmark saved to {args.output}')
    return 0 if all(v['metrics']['successful_runs'] == v['metrics']['attempted_runs']
                    for v in report['variants']) else 1
