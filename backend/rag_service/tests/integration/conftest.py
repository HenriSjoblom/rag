"""
Integration test configuration for RAG service.

This conftest.py provides fixtures for integration tests that test the actual
service components working together, while mocking external service dependencies
like retrieval, generation, and ingestion services.
"""

import shutil
from pathlib import Path

import httpx
import pytest
from app.config import Settings
from app.deps import get_http_client, get_settings
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def integration_test_data_dir() -> Path:
    """Creates a temporary directory for integration test data."""
    path = Path("./test_temp_data/integration")
    if path.exists():
        shutil.rmtree(path)  # Clean up from previous runs
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Cleanup after session
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def integration_settings():
    """Settings for integration tests with mocked service URLs."""
    return Settings(
        RETRIEVAL_SERVICE_URL="http://retrieval-service:8001",
        GENERATION_SERVICE_URL="http://generation-service:8002",
        INGESTION_SERVICE_URL="http://ingestion-service:8004",
        HTTP_CLIENT_TIMEOUT=5.0,
    )


@pytest.fixture
def mock_http_client(mocker):
    """Mock HTTP client for integration tests."""
    mock_client = mocker.MagicMock(spec=httpx.AsyncClient)

    # Configure default successful responses
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.json.return_value = {"status": "ok"}
    mock_response.headers.get.return_value = "application/json"
    mock_response.raise_for_status.return_value = None

    # Use AsyncMock for async methods
    mock_client.request = mocker.AsyncMock(return_value=mock_response)
    mock_client.post = mocker.AsyncMock(return_value=mock_response)
    mock_client.get = mocker.AsyncMock(return_value=mock_response)
    mock_client.delete = mocker.AsyncMock(return_value=mock_response)

    return mock_client


@pytest.fixture
def integration_test_client(integration_settings, mock_http_client):
    """Test client for integration tests with mocked dependencies."""

    # Override dependencies
    app.dependency_overrides[get_settings] = lambda: integration_settings
    app.dependency_overrides[get_http_client] = lambda: mock_http_client

    with TestClient(app) as client:
        yield client

    # Clean up overrides
    app.dependency_overrides.clear()


# Mock response fixtures for different service scenarios
@pytest.fixture
def mock_retrieval_response():
    """Mock response from retrieval service."""
    return {
        "chunks": [
            "This is a sample document chunk about iPhone features.",
            "Another relevant chunk about iPhone functionality.",
            "Additional context about iPhone usage.",
        ]
    }


@pytest.fixture
def mock_generation_response():
    """Mock response from generation service."""
    return {
        "answer": "Based on the provided context, here is a comprehensive answer about iPhone features and functionality."
    }


@pytest.fixture
def mock_ingestion_upload_response():
    """Mock response from ingestion service for document upload."""
    return {
        "message": "File uploaded successfully",
        "filename": "test_document.pdf",
        "status": "ok",
    }


@pytest.fixture
def mock_ingestion_list_response():
    """Mock response from ingestion service for document listing."""
    return {
        "documents": [
            {
                "filename": "iphone_user_guide.pdf",
                "upload_date": "2024-01-15T10:30:00Z",
                "size": 1024000,
                "pages": 45,
            },
            {
                "filename": "test_document.pdf",
                "upload_date": "2024-01-16T14:20:00Z",
                "size": 512000,
                "pages": 23,
            },
        ],
        "count": 2,
    }


@pytest.fixture
def mock_ingestion_status_response():
    """Mock response from ingestion service for status check."""
    return {
        "status": "ready",
        "is_processing": False,
        "total_documents": 2,
        "total_chunks": 150,
        "last_updated": "2024-01-16T14:25:00Z",
    }


@pytest.fixture
def mock_ingestion_delete_response():
    """Mock response from ingestion service for document deletion."""
    return {
        "message": "All documents and ingested data deleted successfully",
        "deleted_documents": 2,
        "deleted_chunks": 150,
    }


@pytest.fixture
def sample_chat_queries():
    """Sample chat queries for testing."""
    return [
        "How do I reset my iPhone?",
        "What are the camera features?",
        "How to set up Face ID?",
        "Battery optimization tips",
        "How to backup iPhone data?",
    ]


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for upload testing."""
    # Mock PDF content as bytes
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000109 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n178\n%%EOF"


@pytest.fixture
def configured_retrieval_client(
    integration_test_client, mock_retrieval_response, mocker
):
    """Test client with configured retrieval service responses."""
    mock_http_client = integration_test_client.app.dependency_overrides[
        get_http_client
    ]()

    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.json.return_value = mock_retrieval_response
    mock_response.raise_for_status.return_value = None

    mock_http_client.request = mocker.AsyncMock(return_value=mock_response)

    return integration_test_client


@pytest.fixture
def configured_generation_client(
    integration_test_client, mock_generation_response, mock_retrieval_response, mocker
):
    """Test client with configured generation service responses."""
    mock_http_client = integration_test_client.app.dependency_overrides[
        get_http_client
    ]()

    # Configure specific responses for different endpoints
    async def request_side_effect(method, url, **kwargs):
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.raise_for_status.return_value = None

        url_str = str(url)

        # Match the exact URL patterns used by the services
        if "retrieval-service:8001" in url_str and "retrieve" in url_str:
            # Retrieval service call
            mock_response.json.return_value = mock_retrieval_response
        elif "generation-service:8002" in url_str and "generate" in url_str:
            # Generation service call
            mock_response.json.return_value = mock_generation_response
        elif "retrieve" in url_str:
            # Fallback for any other retrieval patterns
            mock_response.json.return_value = mock_retrieval_response
        elif "generate" in url_str:
            # Fallback for any other generation patterns
            mock_response.json.return_value = mock_generation_response
        else:
            # Default to retrieval response since chat starts with retrieval
            mock_response.json.return_value = mock_retrieval_response

        return mock_response

    # Override the request method with side effect (this is what make_request actually uses)
    mock_http_client.request = mocker.AsyncMock(side_effect=request_side_effect)

    return integration_test_client


@pytest.fixture
def configured_ingestion_client(
    integration_test_client,
    mock_ingestion_upload_response,
    mock_ingestion_list_response,
    mock_ingestion_status_response,
    mock_ingestion_delete_response,
    mocker,
):
    """Test client with configured ingestion service responses."""
    mock_http_client = integration_test_client.app.dependency_overrides[
        get_http_client
    ]()

    # Configure responses for different endpoints
    def create_mock_response(method, url, **kwargs):
        mock_response = mocker.MagicMock()
        mock_response.is_success = True
        mock_response.raise_for_status.return_value = None

        # Properly mock headers
        mock_response.headers = {"content-type": "application/json"}

        url_str = str(url)
        print(
            f"DEBUG: Mock called with method={method}, url={url_str}"
        )  # Debug logging

        if "upload" in url_str:
            print("DEBUG: Matched upload endpoint")
            mock_response.status_code = 202  # Upload should return 202

            # Extract filename from the uploaded files
            upload_response = mock_ingestion_upload_response.copy()
            if "files" in kwargs and "file" in kwargs["files"]:
                # files is a dict where 'file' is a tuple (filename, content, content_type)
                file_data = kwargs["files"]["file"]
                if isinstance(file_data, tuple) and len(file_data) >= 1:
                    upload_response["filename"] = file_data[0]

            mock_response.json.return_value = upload_response
        elif "status" in url_str:
            print("DEBUG: Matched status endpoint")
            # For status endpoint, simulate service unavailable by raising an exception
            # This will trigger the error handling in the router to return 503
            raise httpx.ConnectError("Mocked ingestion service unavailable")
        elif "documents" in url_str and method.upper() == "GET":
            print("DEBUG: Matched documents GET endpoint")
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ingestion_list_response
        elif "documents" in url_str and method.upper() == "DELETE":
            print("DEBUG: Matched documents DELETE endpoint")
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ingestion_delete_response
        else:
            print("DEBUG: Using fallback response")
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok", "is_processing": False}

        return mock_response

    # Mock the individual HTTP methods that the app actually uses
    mock_http_client.post = mocker.AsyncMock(
        side_effect=lambda url, **kwargs: create_mock_response("POST", url, **kwargs)
    )
    mock_http_client.get = mocker.AsyncMock(
        side_effect=lambda url, **kwargs: create_mock_response("GET", url, **kwargs)
    )
    mock_http_client.delete = mocker.AsyncMock(
        side_effect=lambda url, **kwargs: create_mock_response("DELETE", url, **kwargs)
    )
    # Also override the request method in case it's called directly
    mock_http_client.request = mocker.AsyncMock(
        side_effect=lambda method, url, **kwargs: create_mock_response(
            method, url, **kwargs
        )
    )

    return integration_test_client
