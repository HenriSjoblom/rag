# tests/conftest.py
import pytest
import pytest_asyncio
from typing import Generator, AsyncGenerator
# Import both MagicMock and AsyncMock
from unittest.mock import MagicMock, AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.config import Settings
from app.deps import get_settings, get_generation_service

# -- Settings Override --

@pytest.fixture(scope="session")
def override_settings() -> Settings:
    """Creates a Settings instance for testing, ensuring required keys exist."""
    # Provide dummy values, especially for secrets, to satisfy validation
    return Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL_NAME="test-model",
        LLM_TEMPERATURE=0.1,
        LLM_MAX_TOKENS=50,
        OPENAI_API_KEY="TEST_KEY_DO_NOT_USE", # Dummy key for validation
    )

# -- Mock Service Fixture (for Integration Tests) --

@pytest.fixture(scope="function")
def mock_generation_service() -> MagicMock: # Change return type hint to MagicMock
    """
    Creates a mock GenerationService object (using MagicMock shell)
    suitable for dependency injection in integration tests.
    It mimics the necessary async methods like 'generate_answer'.
    """
    # Use MagicMock as the main container for the service mock
    mock_service = MagicMock(name="MockGenerationService")

    # Attach an AsyncMock specifically for the async 'generate_answer' method
    mock_service.generate_answer = AsyncMock(name="MockGenerateAnswerMethod")

    print("Created mock GenerationService object (MagicMock with AsyncMock method).")
    return mock_service

# -- Integration Test Fixtures --
@pytest_asyncio.fixture(scope="function")
async def test_app(
    override_settings: Settings,
    mock_generation_service: MagicMock # Update type hint here too
) -> AsyncGenerator[FastAPI, None]:
    """
    Creates a test FastAPI app instance with overridden dependencies.
    Injects the mock_generation_service.
    """
    fastapi_app.dependency_overrides[get_settings] = lambda: override_settings
    # Inject the MagicMock object (which has an AsyncMock 'generate_answer' method)
    fastapi_app.dependency_overrides[get_generation_service] = lambda: mock_generation_service

    print("Test App created with overridden dependencies (using mock GenerationService).")
    yield fastapi_app

    fastapi_app.dependency_overrides = {}
    print("Test App dependency overrides cleared.")


@pytest.fixture(scope="function")
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Creates a FastAPI TestClient instance for making requests."""
    with TestClient(test_app) as test_client:
        print("TestClient created.")
        yield test_client
        print("TestClient teardown.")