import pytest
import pytest_asyncio
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
from app.deps import get_settings
from app.services.http_client import lifespan_http_client

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

@pytest_asyncio.fixture(scope="session")
async def test_app() -> FastAPI:
    """
    Provides the FastAPI application instance for testing.
    Lifespan events will be run once per session.
    """
    fastapi_app.dependency_overrides[get_settings] = lambda: override_settings
    print("Applied settings override.")

    # Manually create and run the lifespan manager
    lifespan_manager = lifespan_http_client(
        app=fastapi_app,
        timeout=5,
    )
    print("Created lifespan manager.")

    async with lifespan_manager:
        yield fastapi_app


@pytest_asyncio.fixture(scope="function")
async def client(test_app: FastAPI) -> AsyncGenerator[TestClient, None]:
    """
    A TestClient instance for testing the API.
    Lifespan events are managed by TestClient.
    """
    with TestClient(test_app) as test_client:
        yield test_client