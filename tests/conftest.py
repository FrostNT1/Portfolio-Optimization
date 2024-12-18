import os
import pytest

@pytest.fixture(scope="session")
def test_config_path():
    """Return the path to the test configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "test_config.yaml") 