import time

import pytest

# Variables globales pour suivre le timing des appels API
last_api_call = 0
MIN_DELAY = 1.6  # Minimum delay in seconds


# Créez un marqueur pour les tests LLM
def pytest_configure(config):
    """
    Add a marker for LLM API tests.
    This allows us to easily identify and manage tests that interact with the LLM API.
    """
    config.addinivalue_line("markers", "llm_api: mark a test that calls the LLM API")


@pytest.fixture(autouse=True)
def api_rate_limit(request):
    """Fixture to ensure API rate limits are respected."""
    # N'appliquer le délai qu'aux tests marqués avec llm_api
    if request.node.get_closest_marker("llm_api") is None:
        yield
        return

    global last_api_call

    # Calculate time since last API call
    current_time = time.time()
    elapsed = current_time - last_api_call

    # If not enough time has passed, wait
    if elapsed < MIN_DELAY:
        wait_time = MIN_DELAY - elapsed
        time.sleep(wait_time)

    # Update the last API call time
    last_api_call = time.time()

    yield  # This is where the test runs

    # Update again after the test completes
    last_api_call = time.time()
