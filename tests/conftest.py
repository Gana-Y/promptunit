import pytest

from promptunit import config
from promptunit.stats import stats

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Every test gets its own cache dir and default mode."""
    monkeypatch.delenv("PROMPTUNIT_MODE", raising=False)
    config.set_mode(None)
    config.set_cache_dir(str(tmp_path / "cache"))
    stats.reset()
    yield
    config.set_mode(None)
    config.set_cache_dir(None)
