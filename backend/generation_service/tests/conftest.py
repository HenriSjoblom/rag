# tests/conftest.py
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from app.config import Settings
from app.deps import get_generation_service, get_settings
from app.main import app as fastapi_app
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Creates a Settings instance for testing."""
    return Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL_NAME="test-model",
        LLM_TEMPERATURE=0.1,
        LLM_MAX_TOKENS=50,
        OPENAI_API_KEY="TEST_KEY_DO_NOT_USE",  # Dummy key for validation
    )


@pytest.fixture(scope="function")
def mock_generation_service() -> MagicMock:
    """Creates a mock GenerationService for testing."""
    mock_service = MagicMock(name="MockGenerationService")
    mock_service.generate_answer = AsyncMock(name="MockGenerateAnswerMethod")

    # Add attributes for health checks
    mock_service.rag_chain = MagicMock()
    mock_service.chat_model = MagicMock()
    mock_service.settings = MagicMock()
    mock_service.settings.LLM_PROVIDER = "openai"
    mock_service.settings.LLM_MODEL_NAME = "test-model"

    return mock_service


@pytest_asyncio.fixture(scope="function")
async def test_app(
    test_settings: Settings,
    mock_generation_service: MagicMock,
) -> AsyncGenerator[FastAPI, None]:
    """Creates a test FastAPI app instance with overridden dependencies."""
    fastapi_app.dependency_overrides[get_settings] = lambda: test_settings
    fastapi_app.dependency_overrides[get_generation_service] = (
        lambda: mock_generation_service
    )

    yield fastapi_app

    fastapi_app.dependency_overrides = {}


@pytest.fixture(scope="function")
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Creates a FastAPI TestClient instance for making requests."""
    with TestClient(test_app) as test_client:
        yield test_client
