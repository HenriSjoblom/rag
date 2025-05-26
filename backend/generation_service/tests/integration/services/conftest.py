"""
Service-level conftest for generation service integration tests.

This provides fixtures specifically for testing service components
with real dependencies while mocking external providers.
"""

import pytest
from app.services.generation import GenerationService


@pytest.fixture
def integration_generation_service_with_deps(
    integration_settings, mock_langchain_components
):
    """Real GenerationService with all dependencies for service testing."""
    return GenerationService(settings=integration_settings)


@pytest.fixture
def mock_rag_chain_integration():
    """Mock RAG chain for integration testing with realistic behavior."""
    from unittest.mock import MagicMock

    mock_chain = MagicMock()

    async def mock_ainvoke(input_data):
        query = input_data.get("query", "")
        context = input_data.get("context", "")

        # Simulate various response scenarios
        if "empty" in query.lower():
            return ""
        elif "long" in query.lower():
            return "This is a very long response. " * 50
        elif "special" in query.lower():
            return "Response with émojis 🚀 and special characters: €£¥"
        elif "context" in query.lower() and context:
            return f"Based on the context provided: {context[:100]}..., here is the answer."
        else:
            return f"Generated answer for: {query}"

    mock_chain.ainvoke = mock_ainvoke
    return mock_chain


@pytest.fixture
def sample_requests_for_service():
    """Sample requests specifically for service-level testing."""
    return [
        {"query": "Simple question", "context_chunks": ["Simple context"]},
        {"query": "Question with empty context", "context_chunks": []},
        {
            "query": "Question with multiple contexts",
            "context_chunks": ["Context chunk 1", "Context chunk 2", "Context chunk 3"],
        },
        {
            "query": "Question with long context",
            "context_chunks": ["This is a very long context chunk. " * 100],
        },
        {
            "query": "Question with special characters: émojis 🚀",
            "context_chunks": ["Context with unicode: 你好世界"],
        },
    ]
