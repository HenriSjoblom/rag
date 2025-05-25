"""
Integration test configuration for retrieval service.

This conftest.py provides fixtures for integration tests that test the actual
service components working together, while still mocking external dependencies
like ChromaDB and embedding models.
"""

import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock

import chromadb
import numpy as np
import pytest
from app.config import Settings
from app.deps import (
    get_settings,
    get_vector_search_service,
)
from app.main import app
from app.services.chroma_manager import ChromaClientManager
from app.services.embedding_manager import EmbeddingModelManager
from app.services.vector_search import VectorSearchService
from app.services.vector_store_manager import VectorStoreManager
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="session")
def integration_test_data_dir() -> Path:
    """Creates a temporary directory for integration test data."""
    path = Path("./test_temp_data/integration")
    if path.exists():
        shutil.rmtree(path)  # Clean up from previous runs
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Cleanup after session
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def integration_settings(integration_test_data_dir: Path):
    """Settings for integration tests with realistic configuration."""
    chroma_path = integration_test_data_dir / f"chroma_db_{uuid.uuid4().hex[:8]}"
    collection_name = f"integration_test_collection_{uuid.uuid4().hex[:8]}"

    return Settings(
        EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2",
        CHROMA_COLLECTION_NAME=collection_name,
        CHROMA_MODE="local",
        CHROMA_PATH=str(chroma_path),
        TOP_K_RESULTS=3,
    )


@pytest.fixture
def mock_embedding_model_integration():
    """Mock SentenceTransformer model for integration tests with consistent behavior."""
    mock_model = MagicMock(spec=SentenceTransformer)

    def encode_side_effect(*args, **kwargs):
        input_data = args[0]
        convert_to_numpy = kwargs.get("convert_to_numpy", False)

        # Generate consistent embeddings based on input text
        if isinstance(input_data, str):
            # Single query
            # Use hash of text for consistent embeddings
            hash_val = hash(input_data) % 1000
            embedding = np.random.RandomState(hash_val).rand(384).astype(np.float32)
            return embedding if convert_to_numpy else embedding.tolist()
        else:
            # List of documents
            embeddings = []
            for i, text in enumerate(input_data):
                hash_val = hash(text) % 1000
                embedding = np.random.RandomState(hash_val).rand(384).astype(np.float32)
                embeddings.append(
                    embedding.tolist() if not convert_to_numpy else embedding
                )
            return embeddings

    mock_model.encode.side_effect = encode_side_effect
    return mock_model


@pytest.fixture
def mock_chroma_collection_integration():
    """Mock ChromaDB collection for integration tests with realistic behavior."""
    mock_collection = MagicMock(spec=chromadb.Collection)
    mock_collection.name = "integration_test_collection"
    mock_collection.id = "integration_test_id"

    # Mock data storage
    stored_documents = []
    stored_embeddings = []
    stored_metadata = []
    stored_ids = []

    def add_side_effect(documents=None, embeddings=None, metadatas=None, ids=None):
        if documents:
            stored_documents.extend(documents)
        if embeddings:
            stored_embeddings.extend(embeddings)
        if metadatas:
            stored_metadata.extend(metadatas)
        if ids:
            stored_ids.extend(ids)

    def query_side_effect(query_embeddings=None, n_results=3, **kwargs):
        # Simulate vector search by returning stored documents
        if not stored_documents:
            return {
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]],
            }

        # Return first n_results documents with mock distances
        num_results = min(n_results, len(stored_documents))

        return {
            "ids": [stored_ids[:num_results]],
            "distances": [[0.1 * i for i in range(num_results)]],
            "documents": [stored_documents[:num_results]],
            "metadatas": [
                stored_metadata[:num_results] if stored_metadata else [{}] * num_results
            ],
        }

    def count_side_effect():
        return len(stored_documents)

    def get_side_effect(**kwargs):
        return {
            "ids": stored_ids,
            "documents": stored_documents,
            "metadatas": stored_metadata,
            "embeddings": stored_embeddings,
        }

    mock_collection.add.side_effect = add_side_effect
    mock_collection.query.side_effect = query_side_effect
    mock_collection.count.side_effect = count_side_effect
    mock_collection.get.side_effect = get_side_effect

    # Store references for test access
    mock_collection._stored_documents = stored_documents
    mock_collection._stored_metadata = stored_metadata
    mock_collection._stored_ids = stored_ids

    return mock_collection


@pytest.fixture
def mock_chroma_client_integration(mock_chroma_collection_integration):
    """Mock ChromaDB client for integration tests."""
    mock_client = MagicMock(spec=chromadb.ClientAPI)
    mock_client.get_or_create_collection.return_value = (
        mock_chroma_collection_integration
    )
    mock_client.list_collections.return_value = [mock_chroma_collection_integration]
    return mock_client


@pytest.fixture
def integration_chroma_manager(
    integration_settings, mock_chroma_client_integration, mocker
):
    """Real ChromaClientManager with mocked ChromaDB client."""
    # Mock the actual chromadb.Client creation
    mocker.patch("chromadb.Client", return_value=mock_chroma_client_integration)
    mocker.patch(
        "chromadb.PersistentClient", return_value=mock_chroma_client_integration
    )

    manager = ChromaClientManager(integration_settings)
    return manager


@pytest.fixture
def integration_embedding_manager(
    integration_settings, mock_embedding_model_integration, mocker
):
    """Real EmbeddingModelManager with mocked SentenceTransformer."""
    # Mock SentenceTransformer constructor
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_embedding_model_integration,
    )

    manager = EmbeddingModelManager(integration_settings)
    return manager


@pytest.fixture
def integration_vector_store_manager(
    integration_settings,
    integration_chroma_manager,
    integration_embedding_manager,
    mocker,
):
    """Real VectorStoreManager with mocked dependencies."""
    # Mock the embedding function creation
    mock_embedding_function = Mock()
    mocker.patch(
        "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
        return_value=mock_embedding_function,
    )

    manager = VectorStoreManager(
        integration_settings, integration_chroma_manager, integration_embedding_manager
    )
    return manager


@pytest.fixture
def integration_vector_search_service(
    integration_settings,
    integration_chroma_manager,
    integration_embedding_manager,
    integration_vector_store_manager,
):
    """Real VectorSearchService with all dependencies."""
    return VectorSearchService(
        settings=integration_settings,
        chroma_manager=integration_chroma_manager,
        embedding_manager=integration_embedding_manager,
        vector_store_manager=integration_vector_store_manager,
    )


@pytest.fixture
def integration_test_client(
    integration_settings,
    mock_embedding_model_integration,
    mock_chroma_collection_integration,
    mocker,
):
    """Test client with integration settings and properly mocked external dependencies."""
    # Mock the actual external dependencies
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_embedding_model_integration,
    )
    mocker.patch("chromadb.Client")
    mocker.patch("chromadb.PersistentClient")

    # Mock embedding function creation
    mock_embedding_function = Mock()
    mocker.patch(
        "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
        return_value=mock_embedding_function,
    )

    # CRITICAL: Override both the dependency injection AND the global settings
    # used directly in the lifespan function
    app.dependency_overrides[get_settings] = lambda: integration_settings
    mocker.patch("app.deps.global_settings", integration_settings)
    mocker.patch("app.main.get_settings", return_value=integration_settings)

    with TestClient(app) as client:
        yield client

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def populated_integration_client(
    integration_test_client,
    integration_vector_search_service,
    integration_settings,
    mocker,
):
    """Test client with pre-populated data for testing retrieval."""
    # Ensure settings override is maintained
    app.dependency_overrides[get_settings] = lambda: integration_settings
    mocker.patch("app.deps.global_settings", integration_settings)
    mocker.patch("app.main.get_settings", return_value=integration_settings)

    # Override vector search service dependency to use our real service
    app.dependency_overrides[get_vector_search_service] = (
        lambda settings=None, request=None: integration_vector_search_service
    )

    # Add some test documents to the collection
    test_documents = [
        "The iPhone 16 features advanced camera capabilities with improved low-light performance.",
        "Apple's latest smartphone includes a new action button replacing the mute switch.",
        "The device supports 5G connectivity and has extended battery life compared to previous models.",
        "iOS 18 introduces new AI-powered features and enhanced privacy controls.",
        "The iPhone 16 Pro models include titanium construction and professional video recording features.",
    ]

    # Simulate adding documents to the collection
    collection = integration_vector_search_service.vector_store_manager.get_collection()
    embeddings = [
        integration_vector_search_service.embedding_manager.get_model().encode(doc)
        for doc in test_documents
    ]

    collection.add(
        documents=test_documents,
        embeddings=[
            emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings
        ],
        ids=[f"doc_{i}" for i in range(len(test_documents))],
        metadatas=[
            {"source": f"test_source_{i}.txt"} for i in range(len(test_documents))
        ],
    )

    yield integration_test_client

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def sample_retrieval_queries():
    """Sample queries for testing retrieval functionality."""
    return [
        "iPhone camera features",
        "battery life smartphone",
        "iOS new features",
        "titanium construction phone",
        "5G connectivity mobile device",
    ]


@pytest.fixture
def expected_test_documents():
    """Expected documents that should be available in populated collection."""
    return [
        "The iPhone 16 features advanced camera capabilities with improved low-light performance.",
        "Apple's latest smartphone includes a new action button replacing the mute switch.",
        "The device supports 5G connectivity and has extended battery life compared to previous models.",
        "iOS 18 introduces new AI-powered features and enhanced privacy controls.",
        "The iPhone 16 Pro models include titanium construction and professional video recording features.",
    ]
