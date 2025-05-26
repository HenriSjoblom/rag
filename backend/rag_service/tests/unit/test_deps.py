"""
Unit tests for dependency injection functions in the RAG service.
"""

import httpx
import pytest
from app.config import Settings
from app.deps import get_chat_processor_service, get_http_client, get_settings
from app.services.chat_processor import ChatProcessorService
from fastapi import HTTPException


class TestGetHttpClient:
    """Test cases for get_http_client dependency."""

    def test_get_http_client_returns_client(self):
        """Test that get_http_client returns an AsyncClient."""
        client = get_http_client()
        assert isinstance(client, httpx.AsyncClient)
        assert client.timeout.read == 30.0

    def test_get_http_client_creates_new_instance(self):
        """Test that get_http_client creates a new instance each time."""
        client1 = get_http_client()
        client2 = get_http_client()
        assert client1 is not client2


class TestGetSettings:
    """Test cases for get_settings dependency."""

    def test_get_settings_returns_settings(self, mocker):
        """Test that get_settings returns Settings instance."""
        # Mock the app_settings to avoid using real configuration
        mock_settings = Settings(
            RETRIEVAL_SERVICE_URL="http://test-retrieval",
            GENERATION_SERVICE_URL="http://test-generation",
        )
        mocker.patch("app.deps.app_settings", mock_settings)

        settings = get_settings()
        assert isinstance(settings, Settings)
        assert str(settings.RETRIEVAL_SERVICE_URL) == "http://test-retrieval/"
        assert str(settings.GENERATION_SERVICE_URL) == "http://test-generation/"


class TestGetChatProcessorService:
    """Test cases for get_chat_processor_service dependency."""

    def test_get_chat_processor_service_success(self, mock_http_client):
        """Test successful creation of ChatProcessorService."""
        settings = Settings(
            RETRIEVAL_SERVICE_URL="http://retrieval-service",
            GENERATION_SERVICE_URL="http://generation-service",
        )

        service = get_chat_processor_service(
            settings=settings, http_client=mock_http_client
        )

        assert isinstance(service, ChatProcessorService)
        assert service.retrieval_service_url == "http://retrieval-service/"
        assert service.generation_service_url == "http://generation-service/"
        assert service.http_client == mock_http_client

    def test_get_chat_processor_service_missing_retrieval_url(
        self, mock_http_client, mocker
    ):
        """Test HTTPException when RETRIEVAL_SERVICE_URL is missing."""
        # Mock settings object to have empty/None URL
        mock_settings = mocker.MagicMock()
        mock_settings.RETRIEVAL_SERVICE_URL = None
        mock_settings.GENERATION_SERVICE_URL = "http://generation-service"

        with pytest.raises(HTTPException) as exc_info:
            get_chat_processor_service(
                settings=mock_settings, http_client=mock_http_client
            )

        assert exc_info.value.status_code == 500
        assert (
            "RETRIEVAL_SERVICE_URL or GENERATION_SERVICE_URL is not configured"
            in str(exc_info.value.detail)
        )

    def test_get_chat_processor_service_missing_generation_url(
        self, mock_http_client, mocker
    ):
        """Test HTTPException when GENERATION_SERVICE_URL is missing."""
        # Mock settings object to have empty/None URL
        mock_settings = mocker.MagicMock()
        mock_settings.RETRIEVAL_SERVICE_URL = "http://retrieval-service"
        mock_settings.GENERATION_SERVICE_URL = None

        with pytest.raises(HTTPException) as exc_info:
            get_chat_processor_service(
                settings=mock_settings, http_client=mock_http_client
            )

        assert exc_info.value.status_code == 500
        assert (
            "RETRIEVAL_SERVICE_URL or GENERATION_SERVICE_URL is not configured"
            in str(exc_info.value.detail)
        )

        assert exc_info.value.status_code == 500
        assert (
            "RETRIEVAL_SERVICE_URL or GENERATION_SERVICE_URL is not configured"
            in str(exc_info.value.detail)
        )

    def test_get_chat_processor_service_empty_urls(self, mock_http_client, mocker):
        """Test HTTPException when service URLs are empty strings."""
        # Mock settings object to have empty URLs
        mock_settings = mocker.MagicMock()
        mock_settings.RETRIEVAL_SERVICE_URL = ""
        mock_settings.GENERATION_SERVICE_URL = ""

        with pytest.raises(HTTPException) as exc_info:
            get_chat_processor_service(
                settings=mock_settings, http_client=mock_http_client
            )

        assert exc_info.value.status_code == 500
        assert (
            "RETRIEVAL_SERVICE_URL or GENERATION_SERVICE_URL is not configured"
            in str(exc_info.value.detail)
        )

    def test_get_chat_processor_service_creates_new_instance(self, mock_http_client):
        """Test that get_chat_processor_service creates new instances."""
        settings = Settings(
            RETRIEVAL_SERVICE_URL="http://retrieval-service",
            GENERATION_SERVICE_URL="http://generation-service",
        )

        service1 = get_chat_processor_service(
            settings=settings, http_client=mock_http_client
        )
        service2 = get_chat_processor_service(
            settings=settings, http_client=mock_http_client
        )

        assert service1 is not service2
        assert isinstance(service1, ChatProcessorService)
        assert isinstance(service2, ChatProcessorService)
