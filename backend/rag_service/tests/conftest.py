import pytest
import shutil
import os
from pathlib import Path
import uuid
from typing import Generator, AsyncGenerator, List, Dict, Any, Tuple
import httpx

from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import FastAPI, BackgroundTasks
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.config import Settings
from app.services.chat_processor import (
    ChatProcessorService,
)

# --- Fixtures ---

@pytest.fixture(scope="session")
def override_settings() -> Settings:
    """Provides mock settings with dummy service URLs."""
    return Settings(
        RETRIEVAL_SERVICE_URL="http://mock-retrieval-service",
        GENERATION_SERVICE_URL="http://mock-generation-service",
    )

@pytest.fixture(scope="session")
def mock_http_client() -> MagicMock:
    """Provides a mock httpx.AsyncClient."""
    # Use AsyncMock for async methods if needed, but make_request is patched directly
    return MagicMock(spec=httpx.AsyncClient)



@pytest.fixture(scope="session")
def mocked_chat_service(
    override_settings: Settings,
    mock_http_client: MagicMock,
) -> ChatProcessorService:
    """Provides a ChatProcessorService instance with mocked dependencies."""
    # No make_request patch here
    service = ChatProcessorService(settings=override_settings, http_client=mock_http_client)
    return service

@pytest.fixture(scope="session")
def test_app():
    """
    Provides the FastAPI application instance for testing.
    Lifespan events will be run once per session.
    """
    yield fastapi_app

@pytest.fixture(scope="session") # Changed to session scope for client
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """
    A TestClient instance for testing the API.
    Lifespan events are managed by TestClient.
    """
    with TestClient(test_app) as test_client:
        yield test_client