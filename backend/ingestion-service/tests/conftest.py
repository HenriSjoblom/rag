import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_root() -> Path:
    """Root directory for all test-related temporary data."""
    path = Path("./test_temp_data")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    yield path


@pytest.fixture(scope="session")
def base_settings():
    """Base settings used across all test types."""
    return {
        "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
        "CHUNK_SIZE": 150,
        "CHUNK_OVERLAP": 20,
    }
