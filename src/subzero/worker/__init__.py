"""Subzero background subtitle worker and daemon for Jellyfin and YUCAST."""

from subzero import __version__
from importlib import import_module

__all__ = ["__version__", "create_app", "Config", "Job", "JobStore", "Service"]


def __getattr__(name):
    modules = {'create_app': 'api', 'Config': 'config', 'Job': 'jobs',
               'JobStore': 'jobs', 'Service': 'service'}
    if name not in modules:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{modules[name]}', __name__), name)
    globals()[name] = value
    return value
