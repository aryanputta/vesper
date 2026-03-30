"""
pytest configuration for VESPER backend tests.
Sets asyncio_mode to "auto" so that all async test functions are automatically
treated as coroutines without requiring @pytest.mark.asyncio on each one.
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the asyncio_mode ini option if pytest-asyncio is installed."""
    try:
        import pytest_asyncio  # noqa: F401
    except ImportError:
        return
    # pytest-asyncio ≥ 0.21 respects the ini value set below
    config.addinivalue_line("markers", "asyncio: mark test as asyncio coroutine")
