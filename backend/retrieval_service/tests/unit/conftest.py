"""
Unit test configuration for retrieval service.

This conftest.py provides lightweight fixtures for unit tests that use mocks
instead of real dependencies like ChromaDB.
"""

from unittest.mock import MagicMock

import pytest
from app.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for unit tests."""
    return Settings(
        EMBEDDING_MODEL_NAME="test-model",
        CHROMA_COLLECTION_NAME="test_collection",
        TOP_K_RESULTS=3,
        CHROMA_MODE="local",
        CHROMA_PATH="/tmp/test",
    )


@pytest.fixture
def mock_docker_settings():
    """Mock settings for docker mode testing."""
    return Settings(
        EMBEDDING_MODEL_NAME="test-model",
        CHROMA_COLLECTION_NAME="test_collection",
        TOP_K_RESULTS=3,
        CHROMA_MODE="docker",
        CHROMA_HOST="http://localhost",
        CHROMA_PORT=8000,
    )


@pytest.fixture
def mock_chroma_manager():
    """Mock ChromaClientManager for unit tests."""
    mock_manager = MagicMock()
    mock_client = MagicMock()
    mock_manager.get_client.return_value = mock_client
    return mock_manager


@pytest.fixture
def mock_embedding_manager():
    """Mock EmbeddingModelManager for unit tests."""
    mock_manager = MagicMock()
    mock_model = MagicMock()

    # Set up the mock model to return proper embeddings
    mock_embedding = MagicMock()
    mock_embedding.ndim = 1
    mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.encode.return_value = mock_embedding

    mock_manager.get_model.return_value = mock_model
    return mock_manager


@pytest.fixture
def mock_vector_store_manager():
    """Mock VectorStoreManager for unit tests."""
    mock_manager = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collection.id = "test_id"
    mock_manager.get_collection.return_value = mock_collection
    return mock_manager


@pytest.fixture
def mock_managers(
    mock_settings,
    mock_chroma_manager,
    mock_embedding_manager,
    mock_vector_store_manager,
):
    """Convenience fixture that provides all mock managers."""
    return {
        "settings": mock_settings,
        "chroma_manager": mock_chroma_manager,
        "embedding_manager": mock_embedding_manager,
        "vector_store_manager": mock_vector_store_manager,
    }
