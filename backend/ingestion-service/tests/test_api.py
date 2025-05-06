import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Tuple

from fastapi import status, BackgroundTasks
from fastapi.testclient import TestClient

from app.models import IngestionResponse
from app.services.ingestion_processor import IngestionProcessorService, Settings

from app.deps import get_ingestion_processor_service

pytestmark = pytest.mark.usefixtures("client")

# --- Integration Tests ---

def test_trigger_ingestion_success(
    client_with_backgroundtasks: Tuple[TestClient, MagicMock],
    mocker,
    override_settings: Settings
):
    """Test successful triggering of ingestion via API."""

    client, mock_background_tasks = client_with_backgroundtasks

    # Mock the background task runner function itself
    mock_runner = mocker.patch('app.router.run_ingestion_background')

    # Create a mock file object that behaves like a file
    mock_file = MagicMock(spec=Path)
    mock_file.is_file.return_value = True

    # Mock rglob directly on the Path class within the router's context
    # Ensure it returns a list containing our mock file object
    mock_rglob = mocker.patch('app.router.Path.rglob', return_value=[mock_file])

    # Make the API call
    response = client.post("/api/v1/ingest")

    # Assertions
    assert response.status_code == status.HTTP_202_ACCEPTED, f"Expected 202, got {response.status_code}. Response: {response.text}"
    data = response.json()
    assert data["status"] == "Ingestion task started"

    # Check that the mocked rglob was actually called
    # This assertion now targets the patch on the class method
    mock_rglob.assert_called_once_with('*.*')

    # This assertion relies on the logic correctly counting the mocked file
    assert data["documents_found"] == 1

def test_trigger_ingestion_already_running(client: TestClient):
    """Test attempting to trigger ingestion when it's already running."""
    # Use patch context manager to temporarily set the lock flag
    with patch('app.router.is_ingesting', True):
        response = client.post("/api/v1/ingest")

    assert response.status_code == status.HTTP_409_CONFLICT
    assert "ingestion process is already running" in response.json()["detail"]


def test_trigger_ingestion_source_dir_not_found(client: TestClient, mocker):
    """Test triggering ingestion when the source directory doesn't exist."""

    mock_path_instance = MagicMock(spec=Path)
    # Simulate an error during rglob if the directory doesn't exist
    mock_path_instance.rglob.side_effect = FileNotFoundError("Directory not found")
    mocker.patch('app.router.Path', return_value=mock_path_instance)

    response = client.post("/api/v1/ingest")

    # Expecting 500 because the router catches the exception from rglob
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to check source directory contents" in response.json()["detail"]


def test_trigger_ingestion_source_dir_empty(client: TestClient, mocker):
    """Test triggering ingestion when the source directory is empty."""
    # Mock rglob directly on the Path class to return an empty list
    mock_rglob = mocker.patch('app.router.Path.rglob', return_value=[]) # Simulate finding no files

    response = client.post("/api/v1/ingest")

    # Should return 200 OK in this case, not 202, as no background task is started
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "No documents found"
    assert data["documents_found"] == 0
    mock_rglob.assert_called_once_with('*.*') # Verify rglob was called


def test_health_check(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "ok"

