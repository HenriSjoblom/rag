# tests/conftest.py
import pytest
import pytest_asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import AsyncMock # Use AsyncMock for async methods

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app as fastapi_app # Import your FastAPI app instance
from app.config import Settings
from app.deps import get_settings, get_generation_service # Import dependency getters
from app.services.generation import GenerationService

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

@pytest.fixture(scope="function") # Function scope if mock behavior changes per test
def mock_rag_chain_invoke() -> AsyncMock:
    """Provides a reusable AsyncMock for the rag_chain.ainvoke method."""
    return AsyncMock()

@pytest.fixture(scope="function")
def mock_generation_service(
    override_settings: Settings,
    mock_rag_chain_invoke: AsyncMock # Inject the invoke mock
) -> GenerationService:
    """
    Creates a mock GenerationService instance where the actual LLM call
    (rag_chain.ainvoke) is replaced with an AsyncMock.
    """
    # We initialize a real service to get prompt templates etc.
    # but then patch the critical chain invoke method.
    service_instance = GenerationService(settings=override_settings)
    # Replace the invoke method with our mock
    service_instance.rag_chain.ainvoke = mock_rag_chain_invoke
    print("Created Mock GenerationService with patched rag_chain.ainvoke")
    return service_instance

# -- Integration Test Fixtures --

@pytest_asyncio.fixture(scope="function") # Function scope for clean state
async def test_app(
    override_settings: Settings,
    mock_generation_service: GenerationService # Use the mock service fixture
) -> AsyncGenerator[FastAPI, None]:
    """
    Creates a test FastAPI app instance with overridden dependencies.
    Injects the mock_generation_service.
    """
    # Override dependencies
    fastapi_app.dependency_overrides[get_settings] = lambda: override_settings
    # Override the dependency that provides the GenerationService
    fastapi_app.dependency_overrides[get_generation_service] = lambda: mock_generation_service

    # Note: Lifespan isn't strictly needed here as LLM client init is part of
    # GenerationService, which we are mocking/controlling via fixtures.
    # If main.py had a lifespan, you might need `async with TestClient(...)` approach.
    print("Test App created with overridden dependencies.")
    yield fastapi_app

    # Cleanup: Clear overrides after tests using this fixture are done
    fastapi_app.dependency_overrides = {}
    print("Test App dependency overrides cleared.")


@pytest.fixture(scope="function") # Function scope for test client
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    """Creates a FastAPI TestClient instance for making requests."""
    with TestClient(test_app) as test_client:
        print("TestClient created.")
        yield test_client
        print("TestClient teardown.")