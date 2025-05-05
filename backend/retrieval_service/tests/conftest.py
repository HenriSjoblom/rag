import pytest
import pytest_asyncio
import shutil
from pathlib import Path
import uuid
from typing import Generator, AsyncGenerator, List
import tempfile
import numpy as np

from unittest.mock import MagicMock, AsyncMock
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Collection
from chromadb.api.models.Collection import Collection as ChromaCollectionModel

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

from app.main import app as fastapi_app
from app.config import Settings, settings as app_settings
from app.deps import get_settings, get_vector_search_service
from app.services.vector_search import (
    get_embedding_model,
    get_chroma_collection,
    VectorSearchService,
    lifespan_retrieval_service
)

# -- Fixture Configuration --
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Creates a temporary directory for test data."""
    path = Path("./test_temp_data")
    if path.exists():
        shutil.rmtree(path) # Clean up from previous runs
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created test data dir: {path.resolve()}")
    yield path
    # Teardown: Remove the directory after the test session
    #print(f"Removing test data dir: {path.resolve()}")
    #shutil.rmtree(path)

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
def override_settings(test_chroma_path: str, test_collection_name: str) -> Settings: # Add dependencies
    """Creates a Settings instance for testing using individual fields."""

    # Instantiate Settings using the fields defined in app/config.py
    settings = Settings(
        EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2",
        CHROMA_COLLECTION_NAME=test_collection_name,
        CHROMA_PATH=test_chroma_path,
        TOP_K_RESULTS=3, # Use the correct field name from config.py
    )

    yield settings

# -- Mock Fixtures (for Unit Tests) --
@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Provides a mock SentenceTransformer model for unit tests."""
    mock_model = MagicMock(spec=SentenceTransformer)

    # Define a side effect function to handle different calls to encode
    def encode_side_effect(*args, **kwargs):
        input_data = args[0] # The text or list of texts
        convert_to_numpy = kwargs.get('convert_to_numpy', False)
        convert_to_tensor = kwargs.get('convert_to_tensor', False)

        # Determine if input is a single query or a list of documents
        is_single_query = isinstance(input_data, str)
        num_items = 1 if is_single_query else len(input_data)
        embedding_dim = 384 # Example dimension

        # Generate mock embeddings
        mock_embeddings = [np.random.rand(embedding_dim).astype(np.float32) for _ in range(num_items)]

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
    mock_collection.name = "mock_collection" # Add name attribute used in logging
    mock_collection.query = MagicMock(name="MockQueryMethod")

    # Default query result simulating found documents
    default_query_result = {
        'ids': [['doc_id_1', 'doc_id_2']],
        'embeddings': None,
        'documents': [['mock chunk 1', 'mock chunk 2']],
        'metadatas': [[{'source': 'mock'}, {'source': 'mock'}]],
        'distances': [[0.1, 0.2]]
    }
    # Configure the query method's default return value
    mock_collection.query.return_value = default_query_result

    return mock_collection

@pytest.fixture
def mock_search_service() -> AsyncMock:
    """Provides a mock VectorSearchService for dependency injection."""
    mock = AsyncMock(spec=VectorSearchService)
    # Mock the collection attribute needed in the response
    mock.chroma_collection = MagicMock()
    mock.chroma_collection.name = "mock_test_collection"
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

    # Manually create and run the lifespan manager
    lifespan_manager = lifespan_retrieval_service(
        app=fastapi_app,
        model_name=override_settings.EMBEDDING_MODEL_NAME,
        chroma_path=override_settings.CHROMA_PATH, # Or chroma_settings dict
        collection_name=override_settings.CHROMA_COLLECTION_NAME
    )
    print("Created lifespan manager.")
    try:
        async with lifespan_manager:
            yield fastapi_app
    finally:
        print("Tearing down test_app fixture (session scope)...")
        fastapi_app.dependency_overrides = original_overrides
        print("Restored original dependency overrides.")

@pytest_asyncio.fixture(scope="session") # Depends on test_app to ensure lifespan runs
async def populated_chroma_collection(test_app: FastAPI) -> AsyncGenerator[ChromaCollectionModel, None]:
    """
    Populates the ChromaDB collection AFTER test_app has run the lifespan. Runs once per session.
    """
    print("Entering populated_chroma_collection fixture (session scope)...")
    try:
        # Lifespan ran manually in test_app, so globals should be set
        collection = await get_chroma_collection()
        print(f"Retrieved collection '{collection.name}' for population.")
        embedding_model = await get_embedding_model()


        test_docs = {
            "doc_id_1": "This is the first test document about apples.",
            "doc_id_2": "This is a second document, focusing on oranges.",
            "doc_id_3": "A final document discussing apples and oranges together."
        }
        doc_ids = list(test_docs.keys())
        doc_texts = list(test_docs.values())
        print("Generating embeddings (session scope)...")
        embeddings = embedding_model.encode(doc_texts, convert_to_tensor=False).tolist()
        print("Adding documents (session scope)...")
        collection.add(ids=doc_ids, documents=doc_texts, embeddings=embeddings)
        print(f"Test documents added successfully to '{collection.name}'. Count: {collection.count()}")

        yield collection

    except Exception as e:
        print(f"Error during populated_chroma_collection setup: {e}")
        pytest.fail(f"Failed to setup populated_chroma_collection: {e}")
    finally:
        print("Exiting populated_chroma_collection fixture (session scope).")


@pytest_asyncio.fixture(scope="function")
async def client(populated_chroma_collection: ChromaCollectionModel, test_app: FastAPI) -> AsyncGenerator[TestClient, None]: # Depend on population & test_app
    """
    Creates a TestClient. Ensures collection is populated first (via session fixture).
    TestClient will trigger app's registered lifespan again, but it should be idempotent.
    """
    print(f"DEBUG [client fixture]: Collection '{populated_chroma_collection.name}' is populated (session scope).")

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