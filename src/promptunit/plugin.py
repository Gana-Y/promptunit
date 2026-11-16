"""pytest plugin (auto-registered through the ``pytest11`` entry point)."""
from __future__ import annotations

import pytest

from . import config as settings
from .cache import DiskCache
from .stats import stats


def pytest_addoption(parser):
    group = parser.getgroup("promptunit", "Unit tests for LLM prompts")
    group.addoption(
        "--promptunit-mode",
        choices=settings.MODES,
        default=None,
        help="auto: use cached replies, call the model on a miss (default). "
        "live: always call the model. record: call the model and overwrite the cache. "
        "replay: cache only, fail on a miss (ideal for CI).",
    )
    group.addoption("--promptunit-cache-dir", default=None, help="Where to store cached responses.")


def pytest_configure(config):
    config.addinivalue_line("markers", "prompt: a test that exercises an LLM prompt")
    mode = config.getoption("--promptunit-mode", default=None)
    if mode:
        settings.set_mode(mode)
    cache_dir = config.getoption("--promptunit-cache-dir", default=None)
    if cache_dir:
        settings.set_cache_dir(cache_dir)


def pytest_sessionstart(session):
    stats.reset()


def pytest_terminal_summary(terminalreporter):
    if not stats.calls:
        return
    tr = terminalreporter
    tr.section("promptunit")
    live = stats.calls - stats.cache_hits
    tr.write_line(
        f"mode={settings.get_mode()}  calls={stats.calls} (live {live}, cached {stats.cache_hits})  "
        f"tokens in/out={stats.input_tokens}/{stats.output_tokens}  live latency={stats.latency_s:.2f}s"
    )
    if stats.cost_usd:
        tr.write_line(f"estimated cost: ${stats.cost_usd:.4f}")
    if stats.by_model:
        models = ", ".join(f"{m} x{n}" for m, n in sorted(stats.by_model.items()))
        tr.write_line(f"models: {models}")


@pytest.fixture
def promptunit_stats():
    """Access the live statistics object inside a test."""
    return stats


@pytest.fixture
def promptunit_cache():
    """The on-disk cache (``.clear()``, ``.info()``)."""
    return DiskCache(settings.get_cache_dir())
