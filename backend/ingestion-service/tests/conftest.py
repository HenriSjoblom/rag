import pytest
import shutil
import os
from pathlib import Path
import uuid
from typing import Generator, AsyncGenerator, List, Dict, Any, Tuple

from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import FastAPI, BackgroundTasks
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.config import Settings, settings as app_settings

from app.deps import get_settings, get_ingestion_processor_service
from app.services.ingestion_processor import (
    IngestionProcessorService,
    IngestionStatus,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import chromadb
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# --- Test Directory Setup ---

@pytest.fixture(scope="session")
def test_data_root() -> Path:
    """Root directory for all test-related temporary data."""
    path = Path("./test_temp_data")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created test data root dir: {path.resolve()}")
    yield path

    #print(f"Removing test data root dir: {path.resolve()}")
    #shutil.rmtree(path)

@pytest.fixture(scope="session")
def test_source_dir(test_data_root: Path) -> Path:
    """Creates a temporary directory for source documents."""
    path = test_data_root / "test_documents"
    path.mkdir(parents=True, exist_ok=True)
    # Create a dummy PDF file for testing load
    try:
        dummy_pdf_path = path / "dummy_doc.pdf"
        c = canvas.Canvas(str(dummy_pdf_path), pagesize=letter)
        c.drawString(100, 750, "This is a dummy PDF document for testing.")
        c.save()
        print(f"Created dummy PDF: {dummy_pdf_path}")
    except ImportError:
        print("reportlab not installed, creating dummy text file instead.")
        (path / "dummy_doc.txt").write_text("Dummy text content.")

    return path

@pytest.fixture(scope="session")
def test_chroma_path(test_data_root: Path) -> str:
    """Provides the path for the test ChromaDB persistence."""
    path = test_data_root / "chroma_test_db"
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())

@pytest.fixture(scope="session")
def test_collection_name() -> str:
    """Provides a unique collection name for testing."""
    return f"test_ingest_collection_{uuid.uuid4().hex[:8]}"

# --- Settings Override --

@pytest.fixture(scope="session")
def override_settings(
    test_source_dir: Path,
    test_chroma_path: str,
    test_collection_name: str
) -> Settings:
    """Creates a Settings instance specifically for testing."""
    print(f"Using test source dir: {test_source_dir.resolve()}")
    return Settings(
        SOURCE_DIRECTORY=test_source_dir.resolve(), # Relative to the test root
        EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2",
        CHUNK_SIZE=100, # Smaller chunk size for easier testing
        CHUNK_OVERLAP=20,
        CHROMA_LOCAL_PATH=test_chroma_path, # Use temp chroma path
        CHROMA_COLLECTION_NAME=test_collection_name,
        CLEAN_COLLECTION_BEFORE_INGEST=False
    )

# --- Mock Fixtures (for Unit Tests) --

@pytest.fixture
def mock_directory_loader() -> MagicMock:
    """Mocks the DirectoryLoader."""
    mock = MagicMock()
    # Simulate loading dummy documents
    mock.load.return_value = [
        Document(page_content="This is the first chunk of content.", metadata={"source": "doc1.pdf"}),
        Document(page_content="This is the second chunk, slightly longer.", metadata={"source": "doc2.pdf"}),
    ]
    return mock

@pytest.fixture
def mock_text_splitter() -> MagicMock:
    """Mocks the RecursiveCharacterTextSplitter."""
    mock = MagicMock()
    # Simulate splitting documents into chunks (often just pass through in mock)
    def _mock_split(docs):
        # Simple mock: return 1 chunk per doc, maybe modify metadata slightly
        chunks = []
        for i, doc in enumerate(docs):
            chunks.append(Document(page_content=doc.page_content[:50], metadata={**doc.metadata, "chunk": i}))
        return chunks
    mock.split_documents.side_effect = _mock_split
    return mock

@pytest.fixture
def mock_chroma_vector_store() -> MagicMock:
    """Mocks the LangChain Chroma vector store wrapper."""
    # Use MagicMock as the Langchain Chroma wrapper methods aren't typically async
    mock = MagicMock(spec=Chroma)
    mock._collection = MagicMock(spec=chromadb.Collection) # Mock the underlying collection too if needed
    mock.add_documents = MagicMock() # Mock the method we call
    mock._client = MagicMock(spec=chromadb.ClientAPI) # Mock underlying client if needed for persist etc.
    return mock

@pytest.fixture
def mock_chroma_client() -> MagicMock:
    """Mocks the raw chromadb client."""
    mock = MagicMock(spec=chromadb.ClientAPI)
    mock.delete_collection = MagicMock()
    mock.get_or_create_collection = MagicMock(return_value=MagicMock(spec=chromadb.Collection))
    return mock

# --- Integration Test Fixtures ---

@pytest.fixture
def mock_background_tasks() -> MagicMock:
    """Mocks FastAPI's BackgroundTasks."""
    mock = MagicMock(spec=BackgroundTasks)
    mock.add_task = MagicMock()
    print(f"Created mock_background_tasks with ID: {id(mock)}") # Debug print
    return mock

@pytest.fixture(scope="function")
def client(
    override_settings: Settings,
    mock_background_tasks: MagicMock
) -> Generator[TestClient, None, None]:
    """
    Creates a FastAPI TestClient instance for making requests.
    Handles lifespan events, settings override, singleton patching,
    and BackgroundTasks override.
    """
    # Apply patches
    with patch('app.services.ingestion_processor._embedding_model', None), \
         patch('app.services.ingestion_processor._chroma_client', None), \
         patch('app.services.ingestion_processor._vector_store', None), \
         patch('app.router.is_ingesting', False):

        # Store original overrides
        original_overrides = fastapi_app.dependency_overrides.copy()

        # Apply overrides DIRECTLY before creating TestClient
        fastapi_app.dependency_overrides[get_settings] = lambda: override_settings
        fastapi_app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks
        print(f"Applied overrides to fastapi_app ({id(fastapi_app)}): Settings={id(override_settings)}, BGTasks={id(mock_background_tasks)}")

        # Create TestClient AFTER overrides are applied
        try:
            # Ensure we pass the *exact* app instance we just modified
            with TestClient(fastapi_app) as test_client:
                print(f"TestClient created using app instance ID: {id(test_client.app)}")
                # --- Verify overrides on the client's app instance ---
                bg_override = test_client.app.dependency_overrides.get(BackgroundTasks)
                if bg_override:
                    print(f"Override for BackgroundTasks found on test_client.app. ID of returned mock: {id(bg_override())}")
                else:
                    print("!!! Override for BackgroundTasks NOT FOUND on test_client.app !!!")
                # ---
                yield test_client
                print("TestClient context exit.")
        finally:
            # Restore original overrides
            fastapi_app.dependency_overrides = original_overrides
            print("Restored original dependency overrides.")


@pytest.fixture(scope="function")
def client_with_backgroundtasks(
    override_settings: Settings
    # Remove mock_background_tasks from arguments
) -> Generator[Tuple[TestClient, MagicMock], None, None]: # Change return type hint
    """
    Creates a FastAPI TestClient instance and the BackgroundTasks mock.
    Handles lifespan events, settings override, singleton patching,
    and BackgroundTasks override.
    Yields a tuple: (TestClient, mock_background_tasks)
    """
    # --- Create the mock INSIDE the client fixture ---
    mock_bg_tasks = MagicMock(spec=BackgroundTasks)
    mock_bg_tasks.add_task = MagicMock()
    print(f"[Fixture client] Created mock_bg_tasks with ID: {id(mock_bg_tasks)}")
    # ---

    # Apply patches
    with patch('app.services.ingestion_processor._embedding_model', None), \
         patch('app.services.ingestion_processor._chroma_client', None), \
         patch('app.services.ingestion_processor._vector_store', None), \
         patch('app.router.is_ingesting', False):

        # Store original overrides
        original_overrides = fastapi_app.dependency_overrides.copy()

        # Apply overrides using the locally created mock
        fastapi_app.dependency_overrides[get_settings] = lambda: override_settings
        fastapi_app.dependency_overrides[BackgroundTasks] = lambda: mock_bg_tasks # Use local mock
        print(f"[Fixture client] Applied overrides to fastapi_app ({id(fastapi_app)}): Settings={id(override_settings)}, BGTasks={id(mock_bg_tasks)}")

        # Create TestClient using the app with settings override
        with TestClient(fastapi_app) as test_client:
            print(f"[Fixture client] TestClient created using app instance ID: {id(test_client.app)}")

            # --- Apply BackgroundTasks override AFTER client creation ---
            # Store original overrides from the client's app instance (if any)
            original_client_app_overrides = test_client.app.dependency_overrides.copy()
            # Apply the override using the mock instance created in this fixture
            test_client.app.dependency_overrides[BackgroundTasks] = lambda: mock_bg_tasks
            print(f"[Fixture client] Applied BackgroundTasks override (ID: {id(mock_bg_tasks)}) directly to test_client.app ({id(test_client.app)})")
            # --- Verification ---
            bg_override_check = test_client.app.dependency_overrides.get(BackgroundTasks)
            if bg_override_check and id(bg_override_check()) == id(mock_bg_tasks):
                  print(f"[Fixture client] Verified override on test_client.app. Mock ID: {id(mock_bg_tasks)}")
            else:
                  print("[Fixture client] !!! Override verification FAILED on test_client.app !!!")
            # ---

            # --- Yield the tuple ---
            yield test_client, mock_bg_tasks
            # ---
            print("[Fixture client] TestClient context exit.")