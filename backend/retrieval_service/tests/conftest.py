import shutil
import uuid
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import chromadb
import httpx
import numpy as np
import pytest
import pytest_asyncio
from app.config import Settings
from app.deps import get_settings
from app.main import app as fastapi_app
from app.services.vector_search import VectorSearchService
from chromadb.api.models.Collection import Collection as ChromaCollectionModel
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer


# -- Fixture Configuration --
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Creates a temporary directory for test data."""
    path = Path("./test_temp_data")
    if path.exists():
        shutil.rmtree(path)  # Clean up from previous runs
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created test data dir: {path.resolve()}")
    yield path
    # Teardown: Remove the directory after the test session
    # print(f"Removing test data dir: {path.resolve()}")
    # shutil.rmtree(path)


@pytest.fixture(scope="session")
def test_chroma_path(test_data_dir: Path) -> str:
    """Provides the path for the test ChromaDB persistence."""
    return str(test_data_dir / "chroma_test_db")


@pytest.fixture(scope="session")
def test_collection_name() -> str:
    """Provides a unique collection name for testing."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


# -- Settings Override --
@pytest.fixture(scope="session")
def override_settings(
    test_chroma_path: str, test_collection_name: str
) -> Settings:  # Add dependencies
    """Creates a Settings instance for testing using individual fields."""

    # Instantiate Settings using the fields defined in app/config.py
    settings = Settings(
        EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2",
        CHROMA_COLLECTION_NAME=test_collection_name,
        CHROMA_PATH=test_chroma_path,
        TOP_K_RESULTS=3,  # Use the correct field name from config.py
    )

    yield settings


# -- Mock Fixtures (for Unit Tests) --
@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Provides a mock SentenceTransformer model for unit tests."""
    mock_model = MagicMock(spec=SentenceTransformer)

    # Define a side effect function to handle different calls to encode
    def encode_side_effect(*args, **kwargs):
        input_data = args[0]  # The text or list of texts
        convert_to_numpy = kwargs.get("convert_to_numpy", False)
        convert_to_tensor = kwargs.get("convert_to_tensor", False)

        # Determine if input is a single query or a list of documents
        is_single_query = isinstance(input_data, str)
        num_items = 1 if is_single_query else len(input_data)
        embedding_dim = 384  # Example dimension

        # Generate mock embeddings
        mock_embeddings = [
            np.random.rand(embedding_dim).astype(np.float32) for _ in range(num_items)
        ]

        if convert_to_numpy:
            # _embed_query expects a single numpy array for a single query
            if is_single_query:
                return mock_embeddings[0]
            else:
                # This case shouldn't happen with convert_to_numpy=True in the current code
                # but handle it just in case by returning a list of arrays
                return mock_embeddings
        elif convert_to_tensor is False:
            # add_documents expects a list of lists
            return [emb.tolist() for emb in mock_embeddings]
        else:
            # Handle other cases or raise error if needed
            # For now, return list of lists as a default fallback
            return [emb.tolist() for emb in mock_embeddings]

    # Assign the side effect function to the mock's encode method
    mock_model.encode.side_effect = encode_side_effect

    return mock_model


@pytest.fixture
def mock_chroma_collection() -> MagicMock:
    """Provides a mock ChromaDB Collection object for unit tests."""
    mock_collection = MagicMock(spec=chromadb.Collection)
    mock_collection.name = "mock_collection"  # Add name attribute used in logging
    mock_collection.query = MagicMock(name="MockQueryMethod")

    # Default query result simulating found documents
    default_query_result = {
        "ids": [["doc_id_1", "doc_id_2"]],
        "embeddings": None,
        "documents": [["mock chunk 1", "mock chunk 2"]],
        "metadatas": [[{"source": "mock"}, {"source": "mock"}]],
        "distances": [[0.1, 0.2]],
    }
    # Configure the query method's default return value
    mock_collection.query.return_value = default_query_result

    return mock_collection


@pytest.fixture
def mock_search_service() -> AsyncMock:
    """Provides a mock VectorSearchService for dependency injection."""
    mock = AsyncMock(spec=VectorSearchService)
    # Mock the collection_name attribute needed in the response
    mock.settings = MagicMock()
    mock.settings.CHROMA_COLLECTION_NAME = "mock_test_collection"

    # Mock default search behavior
    mock.search.return_value = [
        {
            "id": "test_doc_1",
            "text": "Test document content about apples",
            "metadata": {"source": "test.pdf"},
            "distance": 0.5,
        }
    ]

    return mock


@pytest.fixture
def error_handling_search_service() -> AsyncMock:
    """Mock service that raises various errors for testing error handling."""
    mock = AsyncMock(spec=VectorSearchService)
    mock.settings = MagicMock()
    mock.settings.CHROMA_COLLECTION_NAME = "error_test_collection"

    # Configure to raise different errors
    mock.search.side_effect = Exception("Database connection failed")

    return mock


# -- Integration Test Fixtures --
@pytest_asyncio.fixture(scope="session")
async def test_app(override_settings: Settings) -> AsyncGenerator[FastAPI, None]:
    """
    Creates a test FastAPI app instance for the session, applies overrides,
    and manages the app's lifespan context.
    """
    print("Setting up test_app fixture (session scope)...")
    # Store original overrides to restore later
    original_overrides = fastapi_app.dependency_overrides.copy()
    fastapi_app.dependency_overrides[get_settings] = lambda: override_settings
    print("Applied settings override.")

    # Use the app's lifespan context manager directly
    print("Starting lifespan context...")
    try:
        async with fastapi_app.router.lifespan_context(fastapi_app):
            yield fastapi_app
    finally:
        print("Tearing down test_app fixture (session scope)...")
        fastapi_app.dependency_overrides = original_overrides
        print("Restored original dependency overrides.")


@pytest_asyncio.fixture(scope="session")  # Depends on test_app to ensure lifespan runs
async def populated_chroma_collection(
    test_app: FastAPI,
) -> AsyncGenerator[ChromaCollectionModel, None]:
    """
    Populates the ChromaDB collection AFTER test_app has run the lifespan. Runs once per session.
    """
    print("Entering populated_chroma_collection fixture (session scope)...")
    try:
        # Lifespan ran manually in test_app, so globals should be set
        print("DEBUG: Attempting to retrieve ChromaDB collection...")
        collection = await get_chroma_collection()
        if collection is None:
            raise RuntimeError("ChromaDB collection not initialized.")
        print(f"DEBUG: Retrieved collection '{collection.name}' for population.")
        print(f"Retrieved collection '{collection.name}' for population.")
        embedding_model = await get_embedding_model()

        test_docs = {
            "doc_id_1": "This is the first test document about apples.",
            "doc_id_2": "This is a second document, focusing on oranges.",
            "doc_id_3": "A final document discussing apples and oranges together.",
        }
        doc_ids = list(test_docs.keys())
        doc_texts = list(test_docs.values())
        print("Generating embeddings (session scope)...")
        embeddings = embedding_model.encode(doc_texts, convert_to_tensor=False).tolist()
        print("Adding documents (session scope)...")
        collection.add(ids=doc_ids, documents=doc_texts, embeddings=embeddings)
        print(
            f"Test documents added successfully to '{collection.name}'. Count: {collection.count()}"
        )

        yield collection

    except Exception as e:
        print(f"Error during populated_chroma_collection setup: {e}")
        pytest.fail(f"Failed to setup populated_chroma_collection: {e}")
    finally:
        print("Exiting populated_chroma_collection fixture (session scope).")


@pytest_asyncio.fixture(scope="function")
async def client(
    populated_chroma_collection: ChromaCollectionModel, test_app: FastAPI
) -> AsyncGenerator[TestClient, None]:  # Depend on population & test_app
    """
    Creates a TestClient. Ensures collection is populated first (via session fixture).
    TestClient will trigger app's registered lifespan again, but it should be idempotent.
    """
    print(
        f"DEBUG [client fixture]: Collection '{populated_chroma_collection.name}' is populated (session scope)."
    )

    with TestClient(test_app) as test_client:
        print("TestClient created (may trigger idempotent lifespan check).")
        yield test_client
        print("TestClient teardown.")


def get_injected_settings(client: TestClient) -> Settings:
    """Retrieves the Settings instance injected via dependency overrides."""
    # Access the override function for get_settings from the app instance
    settings_override_func = client.app.dependency_overrides.get(get_settings)
    if not settings_override_func:
        pytest.fail("Dependency override for get_settings not found in test app.")

    injected_settings = settings_override_func()

    return injected_settings


# -- Fixtures for Docker Mode Integration Tests --


def get_docker_settings_config() -> Settings:
    """
    Provides Settings configured for Docker mode, using values from the global app settings
    (likely from .env or defaults) for host and collection name.
    """
    print("DEBUG: Creating docker_run_settings...")
    return Settings(
        EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2",
        CHROMA_COLLECTION_NAME=f"test_collection_{uuid.uuid4().hex[:8]}",
        CHROMA_MODE="docker",
        CHROMA_HOST="http://localhost:8010",
        TOP_K_RESULTS=3,
    )


@pytest.fixture(scope="session")
def docker_settings() -> (
    Settings
):  # Renamed from user's get_docker_run_settings to be the fixture
    """Fixture providing Settings configured for Docker mode."""
    return get_docker_settings_config()


@pytest_asyncio.fixture(scope="session")
async def docker_test_app(docker_settings: Settings) -> AsyncGenerator[FastAPI, None]:
    """
    Creates a test FastAPI app instance configured for Docker mode for the session,
    applies Docker-specific overrides, and manages the app's lifespan.
    """
    print("Setting up docker_test_app fixture (session scope)...")

    fastapi_app.dependency_overrides[get_settings] = lambda: docker_settings
    print(
        f"Applied docker_run_settings override for docker_test_app. Mode: {docker_settings.CHROMA_MODE}, Host: {docker_settings.CHROMA_HOST}, Collection: {docker_settings.CHROMA_COLLECTION_NAME}"
    )
    print(f"DEBUG: Docker run settings: {docker_settings}")

    lifespan_manager = lifespan_retrieval_service(
        app=fastapi_app,
        model_name=docker_settings.EMBEDDING_MODEL_NAME,
        chroma_mode=docker_settings.CHROMA_MODE,
        chroma_host=docker_settings.CHROMA_HOST,
        collection_name=docker_settings.CHROMA_COLLECTION_NAME,
    )
    print("Created lifespan manager for docker_test_app.")
    async with lifespan_manager:
        print("Lifespan manager context entered for docker_test_app.")
        yield fastapi_app


@pytest.fixture(scope="function")
def docker_client(docker_test_app: FastAPI) -> TestClient:
    """
    Creates a TestClient for interacting with the app configured for Docker mode.
    This client is function-scoped for test isolation.
    """
    print("Creating docker_client (function scope)...")
    with TestClient(docker_test_app) as client:
        print("docker_client created.")
        yield client
        print("docker_client teardown.")


def is_docker_chromadb_available() -> bool:
    """Check if the Docker-based ChromaDB is reachable."""
    # Ensure you are using the correct heartbeat endpoint for your ChromaDB version

    docker_run_settings = get_docker_settings_config()
    heartbeat_url = f"{docker_run_settings.CHROMA_HOST.rstrip('/')}/api/v2/heartbeat"
    print(f"DEBUG: Checking Docker-based ChromaDB availability at {heartbeat_url}...")
    try:
        response = httpx.get(heartbeat_url, timeout=2)
        print(f"DEBUG: Docker-based ChromaDB response status: {response.status_code}")
        return response.status_code == 200
    except httpx.RequestError as e:
        print(f"Request error while connection docker-based ChromaDB: {e}")
        return False
