"""Backward-compatibility shim redirecting to subzero.worker."""
from subzero import worker
import sys

sys.modules["srtworker"] = worker
for k, v in list(worker.__dict__.items()):
    globals()[k] = v
