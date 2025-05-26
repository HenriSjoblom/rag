from unittest.mock import AsyncMock

import pytest
from app.config import Settings
from app.models import GenerateRequest
from app.services.generation import GenerationService


@pytest.fixture
def unit_settings():
    """Settings for unit tests."""
    return Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL_NAME="gpt-3.5-turbo-test",
        LLM_TEMPERATURE=0.7,
        LLM_MAX_TOKENS=500,
        OPENAI_API_KEY="test-key-do-not-use-in-production",
        LOG_LEVEL="DEBUG",
    )


@pytest.fixture
def sample_generate_request():
    """Sample GenerateRequest for testing."""
    return GenerateRequest(
        query="What is FastAPI?",
        context_chunks=[
            "FastAPI is a modern, fast web framework for building APIs with Python."
        ],
    )


@pytest.fixture
def clean_generation_service(unit_settings):
    """Provides a clean GenerationService instance for unit tests."""
    return GenerationService(settings=unit_settings)


@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain for testing."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock()
    return mock_chain


@pytest.fixture
def generation_service_with_mock_chain(
    clean_generation_service, mock_rag_chain, mocker
):
    """Generation service with mocked RAG chain."""
    mocker.patch.object(clean_generation_service, "rag_chain", mock_rag_chain)
    return clean_generation_service, mock_rag_chain


@pytest.fixture
def mocked_generation_service(unit_settings):
    """Provides a GenerationService instance with mocked dependencies for unit tests."""
    return GenerationService(settings=unit_settings)
