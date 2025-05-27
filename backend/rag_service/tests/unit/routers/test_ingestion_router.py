"""
Unit tests for ingestion router endpoints in the RAG service.
"""

import pytest
from app.models import IngestionStatusResponse
from app.routers.ingestion import get_ingestion_status


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
            "last_ingestion_time": "2024-01-01T12:00:00Z",
        }
        mock_http_client.get.return_value = response
        response.raise_for_status = mocker.MagicMock()

        result = await get_ingestion_status(
            http_client=mock_http_client, settings=mock_settings
        )

        assert isinstance(result, IngestionStatusResponse)
        assert result.status == "idle"
        assert result.is_processing is False
        assert result.documents_processed == 10

        mock_http_client.get.assert_called_once_with(
            "http://ingestion:8004/api/v1/status", timeout=30.0
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
            "current_document": "processing_doc.pdf",
        }
        mock_http_client.get.return_value = response
        response.raise_for_status = mocker.MagicMock()

        result = await get_ingestion_status(
            http_client=mock_http_client, settings=mock_settings
        )

        assert result.status == "processing"
        assert result.is_processing is True

    @pytest.mark.asyncio
    async def test_get_status_connection_error(self, mock_http_client, mock_settings):
        """Test status when connection fails."""
        import httpx
        from fastapi import HTTPException

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.get.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await get_ingestion_status(
                http_client=mock_http_client, settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "Cannot connect to Ingestion Service" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_status_timeout(self, mock_http_client, mock_settings):
        """Test status when request times out."""
        import httpx
        from fastapi import HTTPException

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"
        mock_http_client.get.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(HTTPException) as exc_info:
            await get_ingestion_status(
                http_client=mock_http_client, settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "status check timed out" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_status_http_error(self, mock_http_client, mock_settings, mocker):
        """Test status when HTTP error occurs."""
        import httpx
        from fastapi import HTTPException

        mock_settings.INGESTION_SERVICE_URL = "http://ingestion:8004/"

        error_response = mocker.MagicMock()
        error_response.status_code = 500
        mock_http_client.get.side_effect = httpx.HTTPStatusError(
            "Server Error", request=mocker.MagicMock(), response=error_response
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_ingestion_status(
                http_client=mock_http_client, settings=mock_settings
            )

        assert exc_info.value.status_code == 503
        assert "Failed to get status" in exc_info.value.detail
