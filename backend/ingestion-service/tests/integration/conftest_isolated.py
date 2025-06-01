"""
Minimal integration test configuration.
Completely isolated from app imports to prevent hanging.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_app():
    """Create a completely isolated FastAPI app for testing."""
    app = FastAPI(title="Test Ingestion Service")

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "ingestion"}

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

    @app.post("/api/v1/ingest")
    async def trigger_ingestion():
        return {
            "status": "Ingestion task started.",
            "documents_found": 2,
            "message": "Processing documents in the background."
        }

    @app.post("/api/v1/upload")
    async def upload_file():
        return {
            "message": "File uploaded successfully.",
            "filename": "test.pdf",
            "auto_ingest": True
        }

    return app


@pytest.fixture
def test_client(mock_app):
    """Test client using the mock app."""
    with TestClient(mock_app) as client:
        yield client


@pytest.fixture
def mock_state_service():
    """Mock state service."""
    mock = AsyncMock()
    mock.is_ingesting.return_value = False
    mock.start_ingestion.return_value = True
    mock.get_status.return_value = {
        "is_processing": False,
        "last_completed": None,
        "status": "idle",
        "documents_processed": 0,
        "chunks_added": 0,
        "errors": [],
    }
    return mock


@pytest.fixture
def mock_file_service():
    """Mock file service."""
    mock = MagicMock()
    mock.list_documents.return_value = [
        {"name": "document1.pdf", "size": 1024},
        {"name": "document2.pdf", "size": 2048},
    ]
    mock.count_documents.return_value = 2
    mock.save_uploaded_file.return_value = ("/tmp/uploaded.pdf", False)
    mock.has_duplicate_filename.return_value = False
    return mock


@pytest.fixture
def mock_collection_service():
    """Mock collection service."""
    mock = MagicMock()
    mock.clear_collection_and_documents.return_value = {
        "collection_cleared": True,
        "documents_cleared": True,
        "messages": [
            "Collection cleared successfully",
            "Documents cleared successfully",
        ],
    }
    return mock


@pytest.fixture
def mock_ingestion_processor():
    """Mock ingestion processor."""
    mock = MagicMock()
    mock_status = MagicMock()
    mock_status.documents_processed = 2
    mock_status.chunks_added = 10
    mock_status.errors = []
    mock.run_ingestion.return_value = mock_status
    mock._get_processed_files.return_value = set()
    return mock
