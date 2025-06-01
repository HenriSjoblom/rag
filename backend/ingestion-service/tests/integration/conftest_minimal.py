"""
Integration test configuration for ingestion service.
Provides fixtures and test client setup with minimal dependencies.
"""

from pathlib import Path
from unittest.mock import MagicMock
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mock heavy dependencies before importing anything from app
sys.modules['chromadb'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from app.config import Settings


@pytest.fixture
def integration_settings(test_data_root, base_settings, tmp_path):
    """Settings for integration tests with real components."""
    chroma_local_path = tmp_path / "chroma_integration"
    chroma_local_path.mkdir(exist_ok=True)
    # Create a source directory with some test data
    source_dir = test_data_root / "integration"
    source_dir.mkdir(exist_ok=True)

    return Settings(
        **base_settings,
        SOURCE_DIRECTORY=str(source_dir),
        CHROMA_MODE="local",
        CHROMA_PATH=str(chroma_local_path),
        CHROMA_COLLECTION_NAME="test_integration_collection",
        CLEAN_COLLECTION_BEFORE_INGEST=True,  # Clean between integration tests
    )


@pytest.fixture
def mock_app():
    """Create a mock FastAPI app for testing without startup dependencies."""
    app = FastAPI(title="Test Ingestion Service")

    # Add basic health endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "ingestion"}

    # Add status endpoint
    @app.get("/api/v1/status")
    async def get_status():
        return {
            "is_processing": False,
            "status": "idle",
            "last_completed": None,
            "documents_processed": 0,
            "chunks_added": 0,
            "errors": []
        }

    # Add ingestion endpoint
    @app.post("/api/v1/ingest")
    async def trigger_ingestion():
        return {
            "status": "Ingestion task started.",
            "documents_found": 2,
            "message": "Processing documents in the background."
        }

    # Add upload endpoint
    @app.post("/api/v1/upload")
    async def upload_file():
        return {
            "message": "File uploaded successfully.",
            "filename": "test.pdf",
            "auto_ingest": True
        }

    return app


@pytest.fixture
def test_client(mock_app, integration_settings):
    """Basic test client with mocked app and integration settings."""
    with TestClient(mock_app) as client:
        yield client


@pytest.fixture
def mock_state_service(mocker):
    """Mock ingestion state service with common default behavior."""
    mock_service = mocker.AsyncMock()
    mock_service.is_ingesting.return_value = False
    mock_service.start_ingestion.return_value = True
    mock_service.get_status.return_value = {
        "is_processing": False,
        "last_completed": None,
        "status": "idle",
        "documents_processed": 0,
        "chunks_added": 0,
        "errors": [],
    }
    return mock_service


@pytest.fixture
def mock_file_service(mocker):
    """Mock file management service with common default behavior."""
    mock_service = mocker.Mock()
    mock_service.list_documents.return_value = [
        {"name": "document1.pdf", "size": 1024},
        {"name": "document2.pdf", "size": 2048},
    ]
    mock_service.count_documents.return_value = 2
    mock_service.save_uploaded_file.return_value = ("/tmp/uploaded.pdf", False)
    mock_service.has_duplicate_filename.return_value = False
    return mock_service


@pytest.fixture
def mock_collection_service(mocker):
    """Mock collection management service with common default behavior."""
    mock_service = mocker.Mock()
    mock_service.clear_collection_and_documents.return_value = {
        "collection_cleared": True,
        "documents_cleared": True,
        "messages": [
            "Collection cleared successfully",
            "Documents cleared successfully",
        ],
    }
    return mock_service


@pytest.fixture
def mock_ingestion_processor(mocker):
    """Mock ingestion processor with common default behavior."""
    mock_processor = mocker.Mock()
    mock_status = mocker.Mock()
    mock_status.documents_processed = 2
    mock_status.chunks_added = 10
    mock_status.errors = []
    mock_processor.run_ingestion.return_value = mock_status
    mock_processor._get_processed_files.return_value = set()
    return mock_processor


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for integration tests."""
    return "This is a sample PDF document for integration testing. It contains multiple sentences to test document splitting and embedding functionality."


@pytest.fixture
def test_pdf_file(integration_settings, sample_pdf_content):
    """Create a test PDF file for integration tests."""
    pdf_path = Path(integration_settings.SOURCE_DIRECTORY) / "test_document.pdf"

    # Create a simple text file that can act as a mock PDF for testing
    pdf_path.write_text(sample_pdf_content, encoding='utf-8')

    yield pdf_path

    # Cleanup
    if pdf_path.exists():
        pdf_path.unlink()
