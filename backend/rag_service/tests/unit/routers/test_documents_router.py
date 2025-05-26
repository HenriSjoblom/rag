"""
Unit tests for documents router endpoints in the RAG service.
"""

import pytest
from io import BytesIO

from app.models import (
    IngestionUploadResponse,
    RagDocumentListResponse,
    RagDocumentDetail,
    IngestionDeleteResponse,
    IngestionStatusResponse
)
from app.routers.documents import (
    upload_document_for_ingestion,
    list_documents_via_ingestion_service,
    delete_all_documents_and_ingested_data,
    get_ingestion_status
)


class TestUploadDocumentForIngestion:
    """Test cases for upload_document_for_ingestion endpoint."""

    @pytest.fixture
    def mock_upload_file(self, mocker):
        """Create a mock PDF file for testing."""
        file = mocker.MagicMock()
        file.filename = "test.pdf"
        file.content_type = "application/pdf"
        file.read = mocker.AsyncMock(return_value=b"fake pdf content")
        file.close = mocker.AsyncMock()
        return file

    @pytest.mark.asyncio
    async def test_upload_document_success(self, mock_http_client, mock_settings, mock_upload_file, mocker):
        """Test successful document upload."""
        # Setup
        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        # Mock health check response
        health_response = mocker.MagicMock()
        health_response.status_code = 200

        # Mock upload response
        upload_response = mocker.MagicMock()
        upload_response.status_code = 202
        upload_response.headers = {"content-type": "application/json"}
        upload_response.json.return_value = {
            "status": "Upload accepted",
            "message": "File uploaded successfully",
            "documents_found": 1
        }

        mock_http_client.get.return_value = health_response
        mock_http_client.post.return_value = upload_response

        # Execute
        result = await upload_document_for_ingestion(
            file=mock_upload_file,
            http_client=mock_http_client,
            settings=mock_settings
        )

        # Verify
        assert isinstance(result, IngestionUploadResponse)
        assert result.status == "Upload accepted"
        assert result.message == "File uploaded successfully"
        assert result.documents_found == 1

        # Verify health check was called
        mock_http_client.get.assert_called_once_with(
            "http://ingestion:8004/health",
            timeout=10.0
        )

        # Verify upload was called
        mock_http_client.post.assert_called_once()
        assert mock_upload_file.close.called

    @pytest.mark.asyncio
    async def test_upload_document_no_filename(self, mock_http_client, mock_settings, mocker):
        """Test upload with no filename raises HTTPException."""
        from fastapi import HTTPException

        file = mocker.MagicMock()
        file.filename = None

        with pytest.raises(HTTPException) as exc_info:
            await upload_document_for_ingestion(
                file=file,
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 400
        assert "No filename provided" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_document_not_pdf(self, mock_http_client, mock_settings, mocker):
        """Test upload with non-PDF file raises HTTPException."""
        from fastapi import HTTPException

        file = mocker.MagicMock()
        file.filename = "test.txt"

        with pytest.raises(HTTPException) as exc_info:
            await upload_document_for_ingestion(
                file=file,
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 400
        assert "Only PDF documents are allowed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_document_health_check_fails(self, mock_http_client, mock_settings, mock_upload_file):
        """Test upload when health check fails."""
        from fastapi import HTTPException
        import httpx

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await upload_document_for_ingestion(
                file=mock_upload_file,
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "Cannot connect to Ingestion Service" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_document_ingestion_service_error(self, mock_http_client, mock_settings, mock_upload_file, mocker):
        """Test upload when ingestion service returns error."""
        from fastapi import HTTPException
        import httpx

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        # Mock successful health check
        health_response = mocker.MagicMock()
        health_response.status_code = 200
        mock_http_client.get.return_value = health_response

        # Mock upload error response
        error_response = mocker.MagicMock()
        error_response.status_code = 409
        error_response.json.return_value = {"detail": "File already exists"}
        mock_http_client.post.side_effect = httpx.HTTPStatusError(
            "Conflict", request=mocker.MagicMock(), response=error_response
        )

        with pytest.raises(HTTPException) as exc_info:
            await upload_document_for_ingestion(
                file=mock_upload_file,
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_upload_document_non_json_response(self, mock_http_client, mock_settings, mock_upload_file, mocker):
        """Test upload with non-JSON response."""
        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        # Mock health check response
        health_response = mocker.MagicMock()
        health_response.status_code = 200
        mock_http_client.get.return_value = health_response

        # Mock upload response with non-JSON content
        upload_response = mocker.MagicMock()
        upload_response.status_code = 202
        upload_response.headers = {"content-type": "text/plain"}
        mock_http_client.post.return_value = upload_response

        # Execute
        result = await upload_document_for_ingestion(
            file=mock_upload_file,
            http_client=mock_http_client,
            settings=mock_settings
        )

        # Verify fallback response
        assert isinstance(result, IngestionUploadResponse)
        assert result.status == "Upload accepted"
        assert result.message == "File upload accepted by ingestion service"


class TestListDocumentsViaIngestionService:
    """Test cases for list_documents_via_ingestion_service endpoint."""

    @pytest.mark.asyncio
    async def test_list_documents_success(self, mock_http_client, mock_settings, mocker):
        """Test successful document listing."""
        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        # Mock successful response
        response = mocker.MagicMock()
        response.json.return_value = {
            "count": 2,
            "documents": [
                {"name": "doc1.pdf"},
                {"name": "doc2.pdf"}
            ]
        }
        mock_http_client.get.return_value = response
        response.raise_for_status = mocker.MagicMock()

        # Execute
        result = await list_documents_via_ingestion_service(
            http_client=mock_http_client,
            settings=mock_settings
        )

        # Verify
        assert isinstance(result, RagDocumentListResponse)
        assert result.count == 2
        assert len(result.documents) == 2
        assert result.documents[0].name == "doc1.pdf"
        assert result.documents[1].name == "doc2.pdf"

        mock_http_client.get.assert_called_once_with(
            "http://ingestion:8004/api/v1/documents/",
            timeout=30.0
        )

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, mock_http_client, mock_settings, mocker):
        """Test listing when no documents exist."""
        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        response = mocker.MagicMock()
        response.json.return_value = {
            "count": 0,
            "documents": []
        }
        mock_http_client.get.return_value = response
        response.raise_for_status = mocker.MagicMock()

        result = await list_documents_via_ingestion_service(
            http_client=mock_http_client,
            settings=mock_settings
        )

        assert result.count == 0
        assert len(result.documents) == 0

    @pytest.mark.asyncio
    async def test_list_documents_connection_error(self, mock_http_client, mock_settings):
        """Test listing when connection fails."""
        from fastapi import HTTPException
        import httpx

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await list_documents_via_ingestion_service(
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "Cannot connect to Ingestion Service" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_list_documents_timeout(self, mock_http_client, mock_settings):
        """Test listing when request times out."""
        from fastapi import HTTPException
        import httpx

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.get.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(HTTPException) as exc_info:
            await list_documents_via_ingestion_service(
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "not responding" in exc_info.value.detail


class TestDeleteAllDocumentsAndIngestedData:
    """Test cases for delete_all_documents_and_ingested_data endpoint."""

    @pytest.mark.asyncio
    async def test_delete_documents_success(self, mock_http_client, mock_settings, mocker):
        """Test successful document deletion."""
        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        response = mocker.MagicMock()
        response.json.return_value = {
            "message": "Successfully deleted all data",
            "files_deleted_count": 5,
            "collection_deleted": True,
            "source_files_cleared": True
        }
        mock_http_client.delete.return_value = response
        response.raise_for_status = mocker.MagicMock()

        result = await delete_all_documents_and_ingested_data(
            http_client=mock_http_client,
            settings=mock_settings
        )

        assert isinstance(result, IngestionDeleteResponse)
        assert result.message == "Successfully deleted all data"
        assert result.files_deleted_count == 5
        assert result.collection_deleted is True
        assert result.source_files_cleared is True

        mock_http_client.delete.assert_called_once_with(
            "http://ingestion:8004/api/v1/collection/"
        )

    @pytest.mark.asyncio
    async def test_delete_documents_service_unavailable(self, mock_http_client, mock_settings):
        """Test deletion when service is unavailable."""
        from fastapi import HTTPException

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.delete.side_effect = Exception("Service unavailable")

        with pytest.raises(HTTPException) as exc_info:
            await delete_all_documents_and_ingested_data(
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "Failed to connect or communicate" in exc_info.value.detail


class TestGetIngestionStatus:
    """Test cases for get_ingestion_status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_success(self, mock_http_client, mock_settings, mocker):
        """Test successful status retrieval."""
        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        response = mocker.MagicMock()
        response.json.return_value = {
            "status": "idle",
            "is_processing": False,
            "documents_processed": 10,
            "last_ingestion_time": "2024-01-01T12:00:00Z"
        }
        mock_http_client.get.return_value = response
        response.raise_for_status = mocker.MagicMock()

        result = await get_ingestion_status(
            http_client=mock_http_client,
            settings=mock_settings
        )

        assert isinstance(result, IngestionStatusResponse)
        assert result.status == "idle"
        assert result.is_processing is False
        assert result.documents_processed == 10

        mock_http_client.get.assert_called_once_with(
            "http://ingestion:8004/api/v1/status",
            timeout=30.0
        )

    @pytest.mark.asyncio
    async def test_get_status_processing(self, mock_http_client, mock_settings, mocker):
        """Test status when processing is active."""
        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        response = mocker.MagicMock()
        response.json.return_value = {
            "status": "processing",
            "is_processing": True,
            "documents_processed": 5,
            "current_document": "processing_doc.pdf"
        }
        mock_http_client.get.return_value = response
        response.raise_for_status = mocker.MagicMock()

        result = await get_ingestion_status(
            http_client=mock_http_client,
            settings=mock_settings
        )

        assert result.status == "processing"
        assert result.is_processing is True

    @pytest.mark.asyncio
    async def test_get_status_connection_error(self, mock_http_client, mock_settings):
        """Test status when connection fails."""
        from fastapi import HTTPException
        import httpx

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await get_ingestion_status(
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "Cannot connect to Ingestion Service" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_status_timeout(self, mock_http_client, mock_settings):
        """Test status when request times out."""
        from fastapi import HTTPException
        import httpx

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.get.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(HTTPException) as exc_info:
            await get_ingestion_status(
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "status check timed out" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_status_http_error(self, mock_http_client, mock_settings, mocker):
        """Test status when HTTP error occurs."""
        from fastapi import HTTPException
        import httpx

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        error_response = mocker.MagicMock()
        error_response.status_code = 500
        mock_http_client.get.side_effect = httpx.HTTPStatusError(
            "Server Error", request=mocker.MagicMock(), response=error_response
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_ingestion_status(
                http_client=mock_http_client,
                settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "Failed to get status" in exc_info.value.detail
