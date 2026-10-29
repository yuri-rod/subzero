import os
from pathlib import Path
import subprocess
import sys


def test_benchmark_and_subtitle_guards_import_without_optional_worker_dependencies(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'src'
    result = subprocess.run([sys.executable, '-S', '-c', '''
import sys
import subzero.benchmark
from subzero.worker import Config
from subzero.worker.guards import check_excellence_guards
from subzero.worker.srt import same_language
assert same_language('pt-BR', 'por')
assert not same_language('en', 'pt-BR')
assert check_excellence_guards('1\\n00:00:01,000 --> 00:00:02,000\\nOlá.\\n', 'pt-BR').ok
assert 'fastapi' not in sys.modules and 'httpx' not in sys.modules
'''], cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(source)},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_lazy_worker_exports_preserve_public_imports():
    import subzero.worker as worker
    from subzero.worker.api import create_app
    from subzero.worker.config import Config
    from subzero.worker.jobs import Job, JobStore
    from subzero.worker.service import Service
    assert (worker.create_app, worker.Config, worker.Job, worker.JobStore, worker.Service) == (
        create_app, Config, Job, JobStore, Service)
