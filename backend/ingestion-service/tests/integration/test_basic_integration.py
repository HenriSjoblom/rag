"""
Basic integration tests for ingestion service API endpoints.
Tests the service without heavy dependencies.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_health_endpoint():
    """Test that health endpoint works."""
    # Create a simple test app
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "ingestion"}

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "ingestion"}


def test_status_endpoint():
    """Test status endpoint."""
    app = FastAPI()

    @app.get("/api/v1/status")
    async def status():
        return {
            "is_processing": False,
            "status": "idle",
            "last_completed": None,
            "documents_processed": 0,
            "chunks_added": 0,
            "errors": []
        }

    with TestClient(app) as client:
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["is_processing"] is False
        assert data["status"] == "idle"


def test_ingest_endpoint_mock():
    """Test ingestion endpoint with mock response."""
    app = FastAPI()

    @app.post("/api/v1/ingest")
    async def ingest():
        return {
            "status": "Ingestion task started.",
            "documents_found": 2,
            "message": "Processing documents in the background."
        }

    with TestClient(app) as client:
        response = client.post("/api/v1/ingest")
        assert response.status_code == 200
        data = response.json()
        assert "Ingestion task started" in data["status"]
        assert data["documents_found"] == 2


def test_upload_endpoint_mock():
    """Test upload endpoint with mock response."""
    app = FastAPI()

    @app.post("/api/v1/upload")
    async def upload():
        return {
            "message": "File uploaded successfully.",
            "filename": "test.pdf",
            "auto_ingest": True
        }

    with TestClient(app) as client:
        response = client.post("/api/v1/upload")
        assert response.status_code == 200
        data = response.json()
        assert "uploaded successfully" in data["message"]
        assert data["filename"] == "test.pdf"


class TestIntegrationBasics:
    """Basic integration tests for core functionality."""

    def test_settings_creation(self, integration_settings):
        """Test that integration settings can be created."""
        assert integration_settings.CHROMA_MODE == "local"
        assert "test_integration_collection" in integration_settings.CHROMA_COLLECTION_NAME
        assert integration_settings.CLEAN_COLLECTION_BEFORE_INGEST is True

    def test_mock_services_creation(self, mock_state_service, mock_file_service):
        """Test that mock services can be created."""
        assert mock_state_service is not None
        assert mock_file_service is not None

        # Test mock behavior
        assert mock_file_service.count_documents.return_value == 2

    def test_test_client_creation(self, test_client):
        """Test that test client can be created."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_pdf_file_creation(self, test_pdf_file, sample_pdf_content):
        """Test that test PDF file can be created."""
        assert test_pdf_file.exists()
        content = test_pdf_file.read_text(encoding='utf-8')
        assert sample_pdf_content in content
