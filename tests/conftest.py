import pytest

from builder_agent.scan import rows


@pytest.fixture(autouse=True, scope="session")
def _session_row_cache(tmp_path_factory):
    """One row-sample cache per test session: taxi data is hashed once, nothing piles up between runs."""
    mp = pytest.MonkeyPatch()
    mp.setattr(rows, "CACHE_FILE", tmp_path_factory.mktemp("row-cache") / "row_samples.json")
    yield
    mp.undo()
