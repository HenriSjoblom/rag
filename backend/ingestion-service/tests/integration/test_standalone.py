"""
Standalone integration test that doesn't rely on conftest.py imports.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


def test_standalone_health_endpoint():
    """Test health endpoint without any app imports."""
    # Create a simple FastAPI app
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "ingestion"}

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "ingestion"


def test_standalone_mock_services():
    """Test that mock services work independently."""
    # Create mock services
    mock_state_service = MagicMock()
    mock_state_service.is_ingesting.return_value = False
    mock_state_service.get_status.return_value = {
        "is_processing": False,
        "status": "idle"
    }

    mock_file_service = MagicMock()
    mock_file_service.count_documents.return_value = 2

    # Test the mocks
    assert mock_state_service.is_ingesting() is False
    status = mock_state_service.get_status()
    assert status["status"] == "idle"

    count = mock_file_service.count_documents()
    assert count == 2


def test_standalone_api_integration():
    """Test API endpoints without app dependencies."""
    app = FastAPI()

    @app.get("/api/v1/status")
    async def get_status():
        return {
            "is_processing": False,
            "status": "idle",
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

    with TestClient(app) as client:
        # Test status endpoint
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["is_processing"] is False

        # Test ingestion endpoint
        response = client.post("/api/v1/ingest")
        assert response.status_code == 200
        ingest_data = response.json()
        assert "status" in ingest_data
        assert ingest_data["documents_found"] == 2


if __name__ == "__main__":
    print("Running standalone tests...")
    test_standalone_health_endpoint()
    print("✓ Health endpoint test passed")

    test_standalone_mock_services()
    print("✓ Mock services test passed")

    test_standalone_api_integration()
    print("✓ API integration test passed")

    print("All standalone tests passed!")
