"""
Root conftest.py for ingestion service tests.

This provides common fixtures for all test directories.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import sys

import pytest

# Mock heavy dependencies at the root level
sys.modules['chromadb'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

# Now safe to import from app
from app.config import Settings


@pytest.fixture(scope="session")
def test_data_root():
    """Creates a temporary directory for test data that persists across the session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ingestion_test_"))
    yield temp_dir
    # Cleanup after session
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def base_settings():
    """Base settings configuration for tests."""
    return {
        "SOURCE_DIRECTORY": "/tmp/test_docs",
        "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
        "CHUNK_SIZE": 1000,
        "CHUNK_OVERLAP": 150,
        "CHROMA_COLLECTION_NAME": "test_collection",
        "CHROMA_MODE": "local",
        "CHROMA_PATH": "/tmp/test_chroma",
        "MAX_FILE_SIZE_MB": 50,
        "CLEAN_COLLECTION_BEFORE_INGEST": False,
    }
