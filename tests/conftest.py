import pytest

from subzero import compute


@pytest.fixture(autouse=True)
def isolate_compute_policy(request, monkeypatch):
    # Ordinary fixtures must never control the host's real launch agents.
    if request.path.name != "test_compute.py":
        monkeypatch.setattr(compute, "_load_policy", lambda: None)
