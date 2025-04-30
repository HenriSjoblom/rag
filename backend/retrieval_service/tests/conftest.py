import pytest
import pytest_asyncio
import shutil
from pathlib import Path
import uuid
from typing import Generator, AsyncGenerator, List

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
from app.deps import get_settings
from app.services.vector_search import (
    get_embedding_model,
    get_chroma_collection,
    VectorSearchService
)

# -- Fixture Configuration --
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Creates a temporary directory for test data (like ChromaDB)."""
    path = Path("./test_temp_data")
    if path.exists():
        shutil.rmtree(path) # Clean up from previous runs
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created test data dir: {path.resolve()}")
    yield path
    # Teardown: Remove the directory after the test session
    print(f"Removing test data dir: {path.resolve()}")
    shutil.rmtree(path)

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
def override_settings(test_chroma_path: str, test_collection_name: str) -> Settings:
    """Creates a Settings instance specifically for testing."""
    # Override settings to use test paths and potentially different models/config
    return Settings(
        EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2", # Use a real (small) model for integration tests
        TOP_K_RESULTS=3,
        CHROMA_MODE="local",
        CHROMA_LOCAL_PATH=test_chroma_path, # Use the temp path
        CHROMA_COLLECTION_NAME=test_collection_name,
        # CHROMA_HOST=None, # Ensure server settings are None for local mode
        # CHROMA_PORT=None,
        # Override other settings if necessary
    )

# -- Mock Fixtures (for Unit Tests) --
@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Provides a mock SentenceTransformer model."""
    mock_model = MagicMock(spec=SentenceTransformer)
    # Simulate the encode method's behavior
    def _mock_encode(texts, convert_to_numpy=False):
        # Return dummy embeddings (list of lists)
        # Ensure the dimension matches what your Chroma setup might expect,
        # or keep it simple if Chroma is also mocked.
        # Dimension for all-MiniLM-L6-v2 is 384
        if isinstance(texts, str):
             num_texts = 1
        else:
             num_texts = len(texts)
        embeddings = [[float(i) for i in range(384)] for _ in range(num_texts)]
        return embeddings if not convert_to_numpy else MagicMock(tolist=lambda: embeddings) # Simulate tolist() if needed
    mock_model.encode.side_effect = _mock_encode
    return mock_model

@pytest.fixture
def mock_chroma_collection() -> AsyncMock:
    """Provides a mock ChromaDB Collection object."""
    mock_collection = AsyncMock(spec=ChromaCollectionModel)
    mock_collection.name = "mock_collection"

    # Simulate the query method - make it async if your service awaits it
    # Even if the underlying library isn't async, mocking it as async
    # is fine if the calling code uses `await`.
    async def _mock_query(query_embeddings: List[List[float]], n_results: int, include: List[str]):
        print(f"Mock Chroma query called with {len(query_embeddings)} embedding(s), n_results={n_results}")
        # Simulate finding some results
        if query_embeddings[0][0] == 0.0: # Example condition based on dummy embedding
            return {
                'ids': [['id1', 'id2']],
                'embeddings': None,
                'documents': [['mock chunk 1', 'mock chunk 2']],
                'metadatas': [[{'source': 'docA'}, {'source': 'docB'}]],
                'distances': [[0.1, 0.2]]
            }
        else: # Simulate finding nothing
             return {'ids': [[]], 'embeddings': None, 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

    mock_collection.query = _mock_query # Assign the async mock function
    return mock_collection


# -- Integration Test Fixtures --
@pytest_asyncio.fixture(scope="session") # Use async fixture for session scope with async setup
async def test_app(override_settings: Settings) -> AsyncGenerator[FastAPI, None]:
    """Creates a test FastAPI app instance with overridden settings and lifespan."""

    # Override the settings dependency for the entire test app session
    fastapi_app.dependency_overrides[get_settings] = lambda: override_settings

    # Manually run lifespan startup events
    async with httpx.AsyncClient(app=fastapi_app, base_url="http://test") as client:
        # The lifespan context manager from the app should handle startup
        # No need to manually call startup here if lifespan is set correctly
        print("Test App Lifespan Startup completed (via lifespan manager).")
        yield fastapi_app # Provide the app instance to tests
        # Lifespan shutdown should be handled automatically on context exit
        print("Test App Lifespan Shutdown completed (via lifespan manager).")

    # Clear overrides after session
    fastapi_app.dependency_overrides = {}


@pytest_asyncio.fixture(scope="session") # Depends on test_app to ensure lifespan runs
async def populated_chroma_collection(test_app: FastAPI) -> AsyncGenerator[Collection, None]:
    """
    Fixture to get the ChromaDB collection *after* the app's lifespan has
    initialized it and populated it with some test data.
    """
    # Get the dependencies directly from the app's state or re-resolve them
    # This assumes the lifespan correctly populates the global _embedding_model and _chroma_collection
    # A more robust way might involve yielding the collection from the lifespan itself,
    # but this works if lifespan modifies globals or app state accessible here.

    # Let's retrieve the dependencies *after* startup via the dependency injectors
    # We need an actual request context or similar to resolve dependencies usually,
    # but since they are cached by lifespan, we can try accessing them via the getters
    # (This might be slightly brittle depending on FastAPI internals/lifespan implementation)

    try:
        collection = await get_chroma_collection()
        model = await get_embedding_model()
        print(f"Populating Chroma collection: {collection.name}")

        test_docs = {
            "doc_id_1": "This is the first test document about apples.",
            "doc_id_2": "This is a second document, focusing on oranges.",
            "doc_id_3": "A final document discussing apples and oranges together."
        }
        ids = list(test_docs.keys())
        docs = list(test_docs.values())

        # Check if already populated (e.g., if scope='session' and run before)
        if collection.count() == 0:
             print("Collection is empty, adding test documents...")
             embeddings = model.encode(docs, convert_to_numpy=True).tolist()
             collection.add(ids=ids, documents=docs, embeddings=embeddings)
             print(f"Added {len(ids)} documents. Collection count: {collection.count()}")
             # Add a small delay to ensure persistence if needed, though usually not required
             # await asyncio.sleep(0.1)
        else:
             print(f"Collection already populated with {collection.count()} documents.")

        yield collection # Provide the populated collection

    except Exception as e:
        print(f"Error during populated_chroma_collection setup: {e}")
        pytest.fail(f"Failed to setup populated Chroma collection: {e}")


@pytest.fixture(scope="function") # Function scope for test client
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """
    Creates a FastAPI TestClient instance for making requests to the test app.
    """
    with TestClient(test_app) as test_client:
        print("TestClient created.")
        yield test_client
        print("TestClient teardown.")