# tests/services/test_generation.py
import pytest
# We still import mock types from unittest.mock when needed
from unittest.mock import AsyncMock, MagicMock, ANY
from fastapi import HTTPException

# --- Application Imports ---
from app.services.generation import GenerationService, RAG_PROMPT_TEMPLATE
from app.models.llm import GenerateRequest
from app.config import Settings

# Mark tests as async
pytestmark = pytest.mark.asyncio

# --- Fixture for clean service instance ---
@pytest.fixture
def generation_service_instance(override_settings: Settings) -> GenerationService:
    """Provides a clean GenerationService instance for unit tests."""
    return GenerationService(settings=override_settings)

# --- Tests for _format_context (No mocking involved here) ---

@pytest.mark.parametrize(
    "chunks, expected_output",
    [
        ([], "No context provided."),
        (["First chunk."], "First chunk."),
        (["Chunk 1.", "Chunk 2."], "Chunk 1.\n---\nChunk 2."),
        (["Line 1\nLine 2", "Chunk 2"], "Line 1\nLine 2\n---\nChunk 2"),
    ],
    ids=["empty", "single", "multiple", "multiline"]
)
def test_format_context(
    generation_service_instance: GenerationService,
    chunks: list[str],
    expected_output: str
):
    """Tests the _format_context method with various inputs."""
    formatted = generation_service_instance._format_context(chunks)
    assert formatted == expected_output

# --- Tests for generate_answer (using pytest-mock's 'mocker' fixture) ---

async def test_generate_answer_success(
    generation_service_instance: GenerationService,
    mocker
):
    """Tests generate_answer successful path, mocking the LLM chain call."""
    # Arrange
    request = GenerateRequest(query="What is FastAPI?", context_chunks=["FastAPI is a web framework."])
    expected_llm_answer = "FastAPI is indeed a Python web framework known for its speed."
    expected_chain_input = {
        "context": "FastAPI is a web framework.",
        "query": "What is FastAPI?"
    }

    # Use the 'mocker' fixture provided by pytest-mock to patch the object
    mock_ainvoke = mocker.patch.object(
        generation_service_instance.rag_chain, # The object instance to patch
        'ainvoke',                             # The attribute (method) name on the object
        new_callable=AsyncMock,                # Use an AsyncMock for the replacement
        return_value=expected_llm_answer       # Configure the mock's return value
    )

    # Act
    actual_answer = await generation_service_instance.generate_answer(request)

    # Assert
    assert actual_answer == expected_llm_answer
    mock_ainvoke.assert_awaited_once_with(expected_chain_input) # Check the mock was called correctly


async def test_generate_answer_llm_error(
    generation_service_instance: GenerationService,
    mocker
):
    """Tests generate_answer when the LLM chain call raises an exception."""
    # Arrange
    request = GenerateRequest(query="What is FastAPI?", context_chunks=["Context"])
    simulated_error = Exception("Simulated LLM API Error (e.g., Rate Limit)")

    # Use the 'mocker' fixture to patch the object's method to raise an error
    mock_ainvoke = mocker.patch.object(
        generation_service_instance.rag_chain,
        'ainvoke',
        new_callable=AsyncMock,
        side_effect=simulated_error # Configure the mock to raise an error
    )

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await generation_service_instance.generate_answer(request)

    assert exc_info.value.status_code == 503
    assert "Failed to get response from LLM" in exc_info.value.detail
    assert exc_info.value.__cause__ is simulated_error
    mock_ainvoke.assert_awaited_once()