"""
Unit test configuration for RAG service.

This conftest.py provides lightweight fixtures for unit tests that use mocks
instead of real dependencies like HTTP clients and external services.
Uses pytest-mock for consistent mocking approach.
"""

import httpx
import pytest
from app.config import Settings
from app.services.chat_processor import ChatProcessorService


@pytest.fixture
def mock_settings():
    """Mock settings for unit tests."""
    return Settings(
        RETRIEVAL_SERVICE_URL="http://mock-retrieval-service",
        GENERATION_SERVICE_URL="http://mock-generation-service",
        LOG_LEVEL="DEBUG",
    )


@pytest.fixture
def mock_http_client(mocker):
    """Mock httpx.AsyncClient for unit tests using pytest-mock."""
    mock_client = mocker.AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = mocker.AsyncMock()
    mock_client.get = mocker.AsyncMock()
    mock_client.put = mocker.AsyncMock()
    mock_client.delete = mocker.AsyncMock()
    return mock_client


@pytest.fixture
def mock_chat_processor_service(mock_settings, mock_http_client, mocker):
    """Mock ChatProcessorService for unit tests using pytest-mock."""
    mock_service = mocker.AsyncMock(spec=ChatProcessorService)
    mock_service.process = mocker.AsyncMock()
    mock_service._call_retrieval_service = mocker.AsyncMock()
    mock_service._call_generation_service = mocker.AsyncMock()
    return mock_service


@pytest.fixture
def chat_processor_service(mock_settings, mock_http_client):
    """Create real ChatProcessorService with mocked dependencies."""
    return ChatProcessorService(
        retrieval_service_url=mock_settings.RETRIEVAL_SERVICE_URL,
        generation_service_url=mock_settings.GENERATION_SERVICE_URL,
        http_client=mock_http_client,
    )


@pytest.fixture
def mock_dependencies(mock_settings, mock_http_client, mock_chat_processor_service):
    """Convenience fixture that provides all mock dependencies."""
    return {
        "settings": mock_settings,
        "http_client": mock_http_client,
        "chat_processor_service": mock_chat_processor_service,
    }
