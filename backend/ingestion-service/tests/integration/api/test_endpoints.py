import io

import pytest
from app.deps import (
    get_file_management_service,
    get_ingestion_state_service,
    get_settings,
)
from app.main import app
from fastapi.testclient import TestClient


# Create a proper test client setup function to avoid ChromaDB connection issues
def get_test_client(integration_settings, mocker):
    """Create a test client with all external dependencies mocked."""

    # Mock ChromaDB and related managers
    mock_chroma_manager = mocker.Mock()
    mock_chroma_client = mocker.Mock()
    mock_chroma_manager.get_client.return_value = mock_chroma_client

    mock_embedding_manager = mocker.Mock()
    mock_embedding_model = mocker.Mock()
    mock_embedding_manager.get_model.return_value = mock_embedding_model

    mock_vector_store_manager = mocker.Mock()

    # Create mock state service for status endpoint
    mock_state_service = mocker.AsyncMock()
    mock_state_service.get_status.return_value = {
        "is_processing": False,
        "last_completed": None,
        "status": "idle",
        "documents_processed": 0,
        "chunks_added": 0,
        "error_message": None,
    }

    # Create mock for file management service
    mock_file_service = mocker.Mock()
    mock_file_service.list_documents.return_value = {
        "count": 2,
        "documents": [
            {"name": "document1.pdf", "size": 1024, "last_modified": "2023-01-01"},
            {"name": "document2.pdf", "size": 2048, "last_modified": "2023-01-02"},
        ],
    }
    mock_file_service.count_documents.return_value = 2

    # Create mock for ingestion processor
    mock_processor = mocker.AsyncMock()

    # Create mock for collection service
    mock_collection_service = mocker.Mock()
    mock_collection_service.clear_collection_and_documents.return_value = {
        "collection_cleared": True,
        "documents_cleared": True,
        "messages": ["Collection cleared", "Documents cleared"],
    }

    # Create a test client
    with TestClient(app) as client:
        # Directly set app state attributes - this is more reliable than overriding lifespan
        app.state.chroma_manager = mock_chroma_manager
        app.state.embedding_manager = mock_embedding_manager
        app.state.vector_store_manager = mock_vector_store_manager
        app.state.ingestion_state_service = mock_state_service

        # Set up all needed dependency overrides
        app.dependency_overrides[get_settings] = lambda: integration_settings
        app.dependency_overrides[get_ingestion_state_service] = (
            lambda: mock_state_service
        )
        app.dependency_overrides[get_file_management_service] = (
            lambda: mock_file_service
        )

        # Handle optional dependencies
        try:
            from app.deps import get_collection_management_service

            app.dependency_overrides[get_collection_management_service] = (
                lambda: mock_collection_service
            )
        except ImportError:
            pass

        try:
            from app.deps import get_ingestion_processor

            app.dependency_overrides[get_ingestion_processor] = lambda: mock_processor
        except ImportError:
            pass

        # Return both client and mocks for verification
        return client, {
            "state_service": mock_state_service,
            "file_service": mock_file_service,
            "processor": mock_processor,
            "collection_service": mock_collection_service,
        }

    # No need to clear dependency overrides here as it's handled by the context manager


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check_endpoint(self, integration_settings, mocker):
        """Test the health check endpoint."""
        client, _ = get_test_client(integration_settings, mocker)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "ingestion"

    def test_status_endpoint_initial_state(self, integration_settings, mocker):
        """Test the status endpoint returns initial state."""
        client, mocks = get_test_client(integration_settings, mocker)

        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_processing" in data
        assert "last_completed" in data
        assert data["is_processing"] is False
        assert "status" in data


class TestIngestionEndpoints:
    """Test ingestion-related endpoints."""

    def test_trigger_ingestion_success(self, integration_settings, mocker):
        """Test successful ingestion trigger."""
        client, mocks = get_test_client(integration_settings, mocker)

        # Ensure is_processing is correctly mocked
        mocks["state_service"].is_processing.return_value = False

        response = client.post("/api/v1/ingest")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["documents_found"] == 2
        assert "message" in data

        # Verify service calls
        mocks["state_service"].start_ingestion.assert_called_once()
        mocks["processor"].run_ingestion.assert_called_once()

    def test_trigger_ingestion_already_running(self, integration_settings, mocker):
        """Test triggering ingestion when already running."""
        client, mocks = get_test_client(integration_settings, mocker)
        mocks["state_service"].is_processing.return_value = True

        response = client.post("/api/v1/ingest")

        assert response.status_code == 409
        data = response.json()
        assert "running" in data["detail"].lower()

    def test_trigger_ingestion_no_documents(self, integration_settings, mocker):
        """Test triggering ingestion with no documents."""
        client, mocks = get_test_client(integration_settings, mocker)
        mocks["state_service"].is_processing.return_value = False
        mocks["file_service"].count_documents.return_value = 0

        response = client.post("/api/v1/ingest")

        assert response.status_code == 400
        data = response.json()
        assert "no documents found" in data["detail"].lower()


class TestDocumentEndpoints:
    """Test document management endpoints."""

    def test_list_documents_success(self, integration_settings, mocker):
        """Test successful document listing."""
        client, mocks = get_test_client(integration_settings, mocker)

        response = client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["documents"]) == 2
        assert data["documents"][0]["name"] == "document1.pdf"
        assert data["documents"][1]["name"] == "document2.pdf"

    def test_upload_document_success(self, integration_settings, mocker):
        """Test successful document upload."""
        client, mocks = get_test_client(integration_settings, mocker)

        # Add return value for upload function
        mocks["file_service"].upload_document.return_value = {
            "name": "uploaded.pdf",
            "size": 1500,
            "path": "/tmp/uploaded.pdf",
        }

        # Create a test file
        test_content = b"This is a test PDF content"
        test_file = io.BytesIO(test_content)

        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", test_file, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "uploaded.pdf"
        assert data["size"] == 1500
        assert "uploaded successfully" in data["message"]

    def test_upload_document_duplicate(self, integration_settings, mocker):
        """Test uploading duplicate document."""
        client, mocks = get_test_client(integration_settings, mocker)
        mocks["file_service"].has_duplicate_filename.return_value = True

        test_content = b"This is a test PDF content"
        test_file = io.BytesIO(test_content)

        response = client.post(
            "/api/v1/upload",
            files={"file": ("duplicate.pdf", test_file, "application/pdf")},
        )

        assert response.status_code == 409
        data = response.json()
        assert "already exists" in data["detail"].lower()

    def test_upload_document_invalid_type(self, integration_settings, mocker):
        """Test uploading invalid file type."""
        client, mocks = get_test_client(integration_settings, mocker)

        test_content = b"This is not a PDF"
        test_file = io.BytesIO(test_content)

        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", test_file, "text/plain")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "invalid file type" in data["detail"].lower()


class TestCollectionEndpoints:
    """Test collection management endpoints."""

    def test_clear_collection_success(self, integration_settings, mocker):
        """Test successful collection clearing."""
        client, mocks = get_test_client(integration_settings, mocker)

        response = client.delete("/api/v1/collection")

        assert response.status_code == 200
        data = response.json()
        assert data["collection_cleared"] is True
        assert data["documents_cleared"] is True
        assert len(data["messages"]) == 2

    def test_clear_collection_partial_failure(self, integration_settings, mocker):
        """Test partial failure in collection clearing."""
        client, mocks = get_test_client(integration_settings, mocker)

        mocks["collection_service"].clear_collection_and_documents.return_value = {
            "collection_cleared": True,
            "documents_cleared": False,
            "messages": [
                "Collection cleared successfully",
                "Failed to clear documents",
            ],
        }

        response = client.delete("/api/v1/collection")

        assert response.status_code == 207  # Multi-status
        data = response.json()
        assert data["collection_cleared"] is True
        assert data["documents_cleared"] is False


class TestErrorHandling:
    """Test error handling in API endpoints."""

    @pytest.fixture
    def client_with_failing_services(self, integration_settings, mocker):
        """Set up services that raise exceptions."""
        # Mock the manager classes to prevent real connections during startup
        mock_chroma_manager = mocker.Mock()
        mock_embedding_manager = mocker.Mock()
        mock_vector_store_manager = mocker.Mock()

        # Mock services that raise exceptions - using AsyncMock for async methods
        mock_state_service = mocker.AsyncMock()
        mock_state_service.get_status.side_effect = Exception(
            "Database connection failed"
        )

        mock_file_service = mocker.Mock()
        mock_file_service.list_documents.side_effect = Exception("File system error")

        # Mock collection and processor services
        mock_collection_service = mocker.Mock()
        mock_processor = mocker.Mock()

        # Mock the manager constructors to prevent startup issues
        mocker.patch("app.main.ChromaClientManager", return_value=mock_chroma_manager)
        mocker.patch(
            "app.main.EmbeddingModelManager", return_value=mock_embedding_manager
        )
        mocker.patch(
            "app.main.VectorStoreManager", return_value=mock_vector_store_manager
        )
        mocker.patch("app.main.IngestionStateService", return_value=mock_state_service)

        # Import all dependency functions we need to override
        try:
            from app.deps import (
                get_collection_management_service,
                get_ingestion_processor,
            )

            app.dependency_overrides[get_collection_management_service] = (
                lambda **kwargs: mock_collection_service
            )
            app.dependency_overrides[get_ingestion_processor] = (
                lambda **kwargs: mock_processor
            )
        except ImportError:
            # Some dependencies might not exist yet
            pass

        app.dependency_overrides[get_settings] = lambda: integration_settings
        app.dependency_overrides[get_ingestion_state_service] = (
            lambda: mock_state_service
        )
        app.dependency_overrides[get_file_management_service] = (
            lambda **kwargs: mock_file_service
        )

        with TestClient(app) as test_client:
            yield test_client

        app.dependency_overrides.clear()

    def test_status_endpoint_error_handling(self, client_with_failing_services):
        """Test error handling in status endpoint."""
        response = client_with_failing_services.get("/api/v1/status")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_documents_endpoint_error_handling(self, client_with_failing_services):
        """Test error handling in documents endpoint."""
        response = client_with_failing_services.get("/api/v1/documents/")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestEndpointIntegration:
    """Integration tests combining multiple endpoints."""

    def test_workflow_check_status_then_trigger(self, client_with_mocked_services):
        """Test a realistic workflow: check status, then trigger ingestion."""
        client, mocks = client_with_mocked_services

        # First, check initial status
        status_response = client.get("/api/v1/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["is_processing"] is False

        # Check documents are available
        docs_response = client.get("/api/v1/documents")
        assert docs_response.status_code == 200
        docs_data = docs_response.json()
        assert docs_data["count"] == 2

        # Then trigger ingestion
        trigger_response = client.post("/api/v1/ingest")
        assert trigger_response.status_code == 200

    def test_api_endpoint_consistency(self, client_with_mocked_services):
        """Test that all API endpoints follow consistent patterns."""
        client, mocks = client_with_mocked_services

        endpoints_to_test = [
            ("/health", 200),
            ("/api/v1/status", 200),
            ("/api/v1/documents", 200),
        ]

        for endpoint, expected_status in endpoints_to_test:
            response = client.get(endpoint)
            assert response.status_code == expected_status

            # All responses should be JSON
            assert response.headers["content-type"] == "application/json"

            # All responses should be valid JSON
            data = response.json()
            assert isinstance(data, dict)
