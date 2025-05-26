"""
Integration test configuration for generation service.

This conftest.py provides fixtures for integration tests that test the actual
service components working together, while still mocking external dependencies
like LLM providers.
"""

import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock

import pytest
from app.config import Settings
from app.deps import get_generation_service, get_settings
from app.main import app
from app.services.generation import GenerationService
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def integration_test_data_dir() -> Generator[Path, None, None]:
    """Creates a temporary directory for integration test data."""
    path = Path("./test_temp_data/integration")
    if path.exists():
        shutil.rmtree(path)  # Clean up from previous runs
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Cleanup after session
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def integration_settings(integration_test_data_dir: Path):
    """Settings for integration tests with realistic configuration."""
    return Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL_NAME="gpt-4.1-test",
        LLM_TEMPERATURE=0.7,
        LLM_MAX_TOKENS=500,
        OPENAI_API_KEY="test-key-do-not-use-in-production",
        LOG_LEVEL="DEBUG",
    )


@pytest.fixture
def mock_openai_chat_model():
    """Mock OpenAI ChatModel for integration tests with realistic behavior."""
    mock_model = AsyncMock()

    # Mock the async invoke method
    async def mock_ainvoke(input_data):
        # Simulate LLM response based on input
        if isinstance(input_data, dict):
            query = input_data.get("query", "")
            context = input_data.get("context", "")
        else:
            # Handle string inputs or other formats
            query = str(input_data)
            context = ""

        # Create realistic responses based on input
        if "error" in query.lower():
            raise Exception("Simulated LLM error")
        elif "timeout" in query.lower():
            raise TimeoutError("Request timeout")
        elif "rate limit" in query.lower():
            raise Exception("Rate limit exceeded")
        elif "authentication" in query.lower():
            raise Exception("Authentication failed")

        # Generate mock response
        response = f"This is a generated response for the query: '{query[:50]}...'. "
        if context:
            response += (
                "Based on the provided context, I can provide relevant information."
            )
        else:
            response += "No specific context was provided."

        return response

    mock_model.ainvoke = mock_ainvoke
    return mock_model


@pytest.fixture
def mock_langchain_components(mock_openai_chat_model, mocker):
    """Mock LangChain components for integration tests."""
    # Mock ChatOpenAI constructor
    mocker.patch(
        "app.services.generation.ChatOpenAI", return_value=mock_openai_chat_model
    )

    # Mock ChatPromptTemplate
    mock_template = AsyncMock()
    mock_template.from_template.return_value = mock_template
    mocker.patch("app.services.generation.ChatPromptTemplate", mock_template)

    # Mock StrOutputParser
    mock_parser = AsyncMock()
    mocker.patch("app.services.generation.StrOutputParser", return_value=mock_parser)

    # Create the mock chain that will be returned from the pipe operation
    async def mock_chain_ainvoke(input_data):
        # Extract data from the input
        if isinstance(input_data, dict):
            query = input_data.get("query", "")
            context = input_data.get("context", "")
        else:
            query = str(input_data)
            context = ""

        # Create realistic responses based on input
        if "error" in query.lower():
            raise Exception("Simulated LLM error")
        elif "timeout" in query.lower():
            raise TimeoutError("Request timeout")
        elif "rate limit" in query.lower():
            raise Exception("Rate limit exceeded")
        elif "authentication" in query.lower():
            raise Exception("Authentication failed")

        # Generate mock response
        response = f"This is a generated response for the query: '{query[:50]}...'. "
        if context:
            response += (
                "Based on the provided context, I can provide relevant information."
            )
        else:
            response += "No specific context was provided."

        return response

    # Create the final mock chain
    mock_chain = AsyncMock()
    mock_chain.ainvoke = mock_chain_ainvoke

    # Mock the pipe operations to return our mock chain
    # When template | chat_model is called, return a mock that supports | parser
    intermediate_chain = AsyncMock()
    intermediate_chain.__or__ = lambda self, other: mock_chain

    mock_template.__or__ = lambda self, other: intermediate_chain

    return {
        "chat_model": mock_openai_chat_model,
        "template": mock_template,
        "parser": mock_parser,
        "chain": mock_chain,
    }


@pytest.fixture
def integration_generation_service(
    integration_settings, mock_langchain_components, mocker
):
    """Real GenerationService with mocked LangChain dependencies."""
    service = GenerationService(settings=integration_settings)

    # After the service is created, replace its rag_chain with our mock
    async def mock_chain_ainvoke(input_data):
        # Extract data from the input
        if isinstance(input_data, dict):
            query = input_data.get("query", "")
            context = input_data.get("context", "")
        else:
            query = str(input_data)
            context = ""

        # Create realistic responses based on input
        if "error" in query.lower():
            raise Exception("Simulated LLM error")
        elif "timeout" in query.lower():
            raise TimeoutError("Request timeout")
        elif "rate limit" in query.lower():
            raise Exception("Rate limit exceeded")
        elif "authentication" in query.lower():
            raise Exception("Authentication failed")

        # Generate mock response
        response = f"This is a generated response for the query: '{query[:50]}...'. "
        if context:
            response += (
                "Based on the provided context, I can provide relevant information."
            )
        else:
            response += "No specific context was provided."

        return response

    # Replace the rag_chain with our async mock
    mock_chain = AsyncMock()
    mock_chain.ainvoke = mock_chain_ainvoke
    service.rag_chain = mock_chain

    return service


@pytest.fixture
def integration_test_client(
    integration_settings,
    integration_generation_service,
    mock_langchain_components,
    mocker,
):
    """Test client with integration settings and properly mocked external dependencies."""
    # Override the dependency injection and global settings
    app.dependency_overrides[get_settings] = lambda: integration_settings
    app.dependency_overrides[get_generation_service] = (
        lambda: integration_generation_service
    )
    mocker.patch("app.deps.global_settings", integration_settings)

    with TestClient(app) as client:
        yield client

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def integration_test_client_with_service(
    integration_test_client,
    integration_generation_service,
    integration_settings,
    mocker,
):
    """Test client with real GenerationService injected."""
    # Override generation service dependency to use our real service
    app.dependency_overrides[get_generation_service] = (
        lambda: integration_generation_service
    )

    yield integration_test_client, integration_generation_service

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def mock_generation_service_for_api():
    """Mock generation service specifically for API tests."""
    mock_service = AsyncMock()
    mock_service.generate_answer.return_value = "Mocked LLM response for testing"
    mock_service.is_healthy.return_value = True
    return mock_service


@pytest.fixture
def test_client_with_mocks(integration_settings, mock_generation_service_for_api):
    """Test client with mocked dependencies for API testing."""
    app.dependency_overrides[get_settings] = lambda: integration_settings
    app.dependency_overrides[get_generation_service] = (
        lambda: mock_generation_service_for_api
    )

    with TestClient(app) as client:
        yield client, mock_generation_service_for_api

    app.dependency_overrides.clear()


@pytest.fixture
def sample_generation_requests():
    """Sample generation requests for testing."""
    return [
        {
            "query": "What is FastAPI?",
            "context_chunks": [
                "FastAPI is a modern, fast web framework for building APIs with Python."
            ],
        },
        {
            "query": "How do I install Python packages?",
            "context_chunks": [
                "You can install Python packages using pip.",
                "The command is: pip install package_name",
            ],
        },
        {
            "query": "Explain machine learning",
            "context_chunks": [
                "Machine learning is a subset of artificial intelligence.",
                "It involves training algorithms on data to make predictions.",
                "Common types include supervised and unsupervised learning.",
            ],
        },
    ]


@pytest.fixture
def large_context_chunks():
    """Large context chunks for testing edge cases."""
    return [f"This is a very long context chunk number {i}. " * 100 for i in range(20)]


@pytest.fixture
def large_context_data():
    """Large context data for testing with substantial content."""
    return [
        f"This is context chunk {i} containing detailed information about device features and capabilities. "
        * 20
        for i in range(15)
    ]


@pytest.fixture
def special_character_data():
    """Test data with special characters and unicode for API tests."""
    return {
        "query": "Comment utiliser les émojis 🚀 et caractères spéciaux: @#$%^&*()?",
        "context_chunks": [
            "Support Unicode: 你好世界 नमस्ते العالم",
            "Émojis: 😊🎉🔥💯📱💻🌟",
            "Symboles spéciaux: €£¥$¢¿¡",
            "Mathématiques: ∑∏∆∇∂∞±≤≥",
            "Quotes: 'single' \"double\" `backtick`",
        ],
    }
