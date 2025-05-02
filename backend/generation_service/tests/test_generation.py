import pytest
from unittest.mock import AsyncMock, ANY # Only need AsyncMock and ANY now
from fastapi import HTTPException, status # Import status

from app.services.generation import GenerationService
from app.models import GenerateRequest
from app.config import Settings


@pytest.fixture
def override_settings() -> Settings:
    """Provides custom Settings for tests."""
    return Settings(
        # Define your test-specific settings here
        EMBEDDING_MODEL_NAME="test-model",
        TOP_K_RESULTS=3,
        CHROMA_MODE="local",
        CHROMA_LOCAL_PATH="./test_data/chroma_db",
        CHROMA_COLLECTION_NAME="test_collection",
        CHROMA_HOST=None,
        CHROMA_PORT=None
    )


# --- Fixture for clean service instance ---
@pytest.fixture
def generation_service_instance(override_settings: Settings) -> GenerationService:
    """Provides a clean GenerationService instance for unit tests."""
    return GenerationService(settings=override_settings)

# --- Tests for _format_context ---
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

# --- Tests for generate_answer ---
@pytest.mark.asyncio
async def test_generate_answer_success(
    generation_service_instance: GenerationService,
    mocker
):
    """Tests generate_answer successful path, mocking the LLM chain call."""

    request = GenerateRequest(query="What is FastAPI?", context_chunks=["FastAPI is a web framework."])
    expected_llm_answer = "FastAPI is indeed a Python web framework known for its speed."
    formatted_context = generation_service_instance._format_context(request.context_chunks)
    expected_chain_input = {
        "context": formatted_context,
        "query": "What is FastAPI?"
    }

    # Create an AsyncMock to replace the rag_chain instance attribute.
    mock_chain_replacement = AsyncMock(name="MockedRagChainInstance")

    # Configure its 'ainvoke' method to be an AsyncMock that returns the desired value.
    mock_chain_replacement.ainvoke = AsyncMock(name="MockedAinvokeMethod", return_value=expected_llm_answer)

    #Patch the 'rag_chain' attribute *on the specific instance*.
    mocker.patch.object(
        generation_service_instance,
        'rag_chain',
        new=mock_chain_replacement
    )

    actual_answer = await generation_service_instance.generate_answer(request)

    assert actual_answer == expected_llm_answer

    #Assert that the ainvoke method on our mock was called correctly
    generation_service_instance.rag_chain.ainvoke.assert_awaited_once_with(expected_chain_input)


@pytest.mark.asyncio
async def test_generate_answer_llm_failure(
    generation_service_instance: GenerationService,
    mocker
):
    """Tests generate_answer when the LLM chain call fails."""
    request = GenerateRequest(query="Test Query", context_chunks=["Some context"])
    expected_chain_input = {
        "context": "Some context",
        "query": "Test Query"
    }
    error_message = "LLM unavailable"

    # Create mock chain that raises an exception on ainvoke
    mock_chain_replacement = AsyncMock(name="FailingMockChain")
    mock_chain_replacement.ainvoke = AsyncMock(name="FailingAinvoke", side_effect=RuntimeError(error_message))

    # Patch the instance attribute
    mocker.patch.object(
        generation_service_instance,
        'rag_chain',
        new=mock_chain_replacement
    )

    # Assert that calling the method raises the expected HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await generation_service_instance.generate_answer(request)

    # Check the details of the raised exception
    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert error_message in exc_info.value.detail

    # Assert the mock was called
    generation_service_instance.rag_chain.ainvoke.assert_awaited_once_with(expected_chain_input)


