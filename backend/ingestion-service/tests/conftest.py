import pytest
import pytest_asyncio
import shutil
import os
from pathlib import Path
import uuid
from typing import Generator, AsyncGenerator, List, Dict, Any

from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import FastAPI, BackgroundTasks
from fastapi.testclient import TestClient
import httpx

from app.main import app as fastapi_app
from app.config import Settings, settings as app_settings
from app.deps import get_settings, get_ingestion_processor_service
from app.services.ingestion_processor import (
    IngestionProcessorService,
    get_embedding_model,
    get_chroma_client,
    get_vector_store,
    IngestionStatus,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

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
    path = test_data_root / "documents"
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

# -- Settings Override --

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
        CHROMA_MODE="local",
        CHROMA_LOCAL_PATH=test_chroma_path, # Use temp chroma path
        CHROMA_COLLECTION_NAME=test_collection_name,
        CLEAN_COLLECTION_BEFORE_INGEST=False
    )

# -- Mock Fixtures (for Unit Tests) --

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

# -- Integration Test Fixtures --

@pytest_asyncio.fixture(scope="function") # Function scope for clean state between tests
async def test_app(
    override_settings: Settings
) -> AsyncGenerator[FastAPI, None]:
    """
    Creates a test FastAPI app instance with overridden settings and lifespan.
    Uses real ChromaDB and Embedding model for integration testing the flow.
    """
    # Reset global singletons in ingestion_processor before each test run
    # This ensures that settings changes between tests are reflected
    with patch('app.services.ingestion_processor._embedding_model', None), \
         patch('app.services.ingestion_processor._chroma_client', None), \
         patch('app.services.ingestion_processor._vector_store', None):

        # Override the settings dependency
        fastapi_app.dependency_overrides[get_settings] = lambda: override_settings

        # Use FastAPI's lifespan context manager via httpx client
        # This ensures startup/shutdown events (like pre-loading) run
        async with httpx.AsyncClient(app=fastapi_app, base_url="http://test") as client:
            print("Test App Lifespan Startup completed (via lifespan manager).")
            yield fastapi_app # Provide the app instance to tests
            print("Test App Lifespan Shutdown completed (via lifespan manager).")

    # Clear overrides after test function
    fastapi_app.dependency_overrides = {}


@pytest.fixture(scope="function")
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Creates a FastAPI TestClient instance for making requests."""
    # Reset the ingestion lock before each test that uses the client
    with patch('app.api.routers.ingest.is_ingesting', False):
        with TestClient(test_app) as test_client:
            print("TestClient created.")
            yield test_client
            print("TestClient teardown.")

@pytest.fixture
def mock_background_tasks() -> MagicMock:
    """Mocks FastAPI's BackgroundTasks."""
    mock = MagicMock(spec=BackgroundTasks)
    mock.add_task = MagicMock()
    return mock

