import pytest
import pytest_asyncio
import shutil
from pathlib import Path
import uuid
from typing import Generator, AsyncGenerator, List
import tempfile

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
    # Create a mock with the spec of the real class for better type hinting/attribute checking
    mock_model = MagicMock(spec=SentenceTransformer)

    # Mock the object that the 'encode' method should return
    mock_encode_result = MagicMock(name="EncodeResultMock")
    # Set the 'ndim' attribute that the code checks
    mock_encode_result.ndim = 1 # Simulate a 1D array by default
    # Set a default return value for the 'tolist' method called later
    mock_encode_result.tolist.return_value = [[0.0] * 384] # Example 384-dim embedding

    # Configure the 'encode' method on the main mock to return our prepared result mock
    mock_model.encode.return_value = mock_encode_result

    return mock_model

@pytest.fixture
def mock_chroma_collection() -> AsyncMock:
    """Provides a mock ChromaDB Collection object for unit tests."""
    # Use AsyncMock because the service awaits collection.query
    mock_collection = AsyncMock(spec=chromadb.Collection)
    mock_collection.name = "mock_collection" # Add name attribute used in logging
    mock_collection.query = AsyncMock(name="MockQueryMethod")

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

@pytest_asyncio.fixture(scope="function") # Depends on test_app to ensure lifespan runs
async def populated_chroma_collection(test_app: FastAPI) -> AsyncGenerator[ChromaCollectionModel, None]:
    """
    Fixture to get the ChromaDB collection *after* the app's lifespan has
    initialized it and populate it with some test data.
    Depends on 'test_app' to ensure lifespan startup is complete.
    """
    print("Entering populated_chroma_collection fixture...")
    try:
        # Retrieve the collection instance created during app lifespan
        collection = await get_chroma_collection()
        print(f"Retrieved collection '{collection.name}' for population.")
        # Retrieve the embedding model instance as well, needed for .add()
        embedding_model = await get_embedding_model()

        # Define test documents
        test_docs = {
            "doc_id_1": "This is the first test document about apples.",
            "doc_id_2": "This is a second document, focusing on oranges.",
            "doc_id_3": "A final document discussing apples and oranges together."
        }
        doc_ids = list(test_docs.keys())
        doc_texts = list(test_docs.values())

        # Generate embeddings for the test documents
        print("Generating embeddings for test documents...")
        embeddings = embedding_model.encode(doc_texts, convert_to_tensor=False).tolist() # Ensure list output
        print(f"Generated {len(embeddings)} embeddings.")

        # Add documents to the collection
        print(f"Adding {len(doc_ids)} documents to collection '{collection.name}'...")
        collection.add(
            ids=doc_ids,
            documents=doc_texts,
            embeddings=embeddings # Provide pre-computed embeddings

        )
        print("Test documents added successfully.")

        # Yield the populated collection so tests can potentially use it
        yield collection

    except Exception as e:
        print(f"Error during populated_chroma_collection setup: {e}")
        pytest.fail(f"Failed to setup populated_chroma_collection: {e}")
    finally:
        # Optional: Cleanup if needed, though collection deletion might happen elsewhere
        # or be handled by deleting the temp directory in test_data_dir
        print("Exiting populated_chroma_collection fixture.")


@pytest_asyncio.fixture(scope="function") # Function scope is typical for clients
async def client(test_app: FastAPI) -> AsyncGenerator[TestClient, None]:
    """
    Creates a FastAPI TestClient instance for making requests to the test app.
    Depends on 'test_app' to ensure the app and its lifespan are ready.
    """

    try:
        # Get the override function for get_settings from the app instance
        settings_override_func = test_app.dependency_overrides.get(get_settings)
        if settings_override_func:
            # Call the override function to get the actual Settings object
            current_settings = settings_override_func()
            print(f"DEBUG [client fixture]: test_app CHROMA_PATH from override: '{current_settings.CHROMA_PATH}'")
            print(f"DEBUG [client fixture]: test_app COLLECTION_NAME from override: '{current_settings.CHROMA_COLLECTION_NAME}'")
        else:
            print("DEBUG [client fixture]: No override found for get_settings on test_app.")
            # Optionally, get default settings if needed for comparison
            # from app.config import settings as default_settings
            # print(f"DEBUG [client fixture]: Default CHROMA_PATH: '{default_settings.CHROMA_PATH}'")

        # You can still check the globally initialized collection name as before
        collection = await get_chroma_collection()
        print(f"DEBUG [client fixture]: Global collection name BEFORE TestClient creation: '{collection.name}'")
    except Exception as e:
        print(f"DEBUG [client fixture]: Error inspecting settings/collection: {e}")


    # Create the TestClient using the app instance provided by the test_app fixture
    with TestClient(test_app) as test_client:
        print("TestClient created.")
        yield test_client # Provide the client to the test function
        print("TestClient teardown.")