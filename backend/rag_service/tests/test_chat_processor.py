import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import List, Dict, Any

from fastapi import HTTPException, status

from app.services.chat_processor import ChatProcessorService
from app.services.http_client import make_request
from app.config import Settings
from app.models import (
    RetrievalRequest,
    RetrievalResponse,
    GenerationRequest,
    GenerationResponse,
    ChatRequest,
)


# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

# --- Unit Tests ---

# Test _call_retrieval_service
async def test_call_retrieval_service_success(
    mocked_chat_service: ChatProcessorService,
    override_settings: Settings,
    mocker
):
    """Test successful call to the retrieval service."""
    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)

    expected_chunks = ["chunk 1", "chunk 2"]

    mock_response_payload = RetrievalResponse(chunks=expected_chunks).model_dump()
    mock_make_request.return_value = mock_response_payload

    print(f"mock_make_request id: {id(mock_make_request)}")

    query = "test query"
    retrieved_chunks = await mocked_chat_service._call_retrieval_service(query)

    expected_url = f"{str(override_settings.RETRIEVAL_SERVICE_URL)}retrieve"
    expected_call_payload = RetrievalRequest(query=query).model_dump()
    print(f"expected_call_payload: {expected_call_payload}")
    print(f"expected_url: {expected_url}")

    mock_make_request.assert_awaited_once_with(
        client=mocked_chat_service.http_client,
        method="POST",
        url=expected_url,
        json_data=expected_call_payload
    )
    assert retrieved_chunks == expected_chunks

async def test_call_retrieval_service_http_error(
    mocked_chat_service: ChatProcessorService,
    mocker
):
    """Test handling HTTP error from retrieval service."""
    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)

    original_exception = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    mock_make_request.side_effect = original_exception

    query = "test query"
    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service._call_retrieval_service(query)

    mock_make_request.assert_awaited_once() # Check it was awaited
    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Retrieval service failed: Not Found" in exc_info.value.detail

async def test_call_retrieval_service_unexpected_error(
    mocked_chat_service: ChatProcessorService,
    mocker
):
    """Test handling unexpected error during retrieval call."""
    # Correctly patch 'make_request' where it's used by ChatProcessorService
    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)
    mock_make_request.side_effect = Exception("Something broke")

    query = "test query"
    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service._call_retrieval_service(query)

    mock_make_request.assert_awaited_once() # Check it was awaited
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "unexpected error occurred while retrieving context" in exc_info.value.detail

async def test_call_retrieval_service_invalid_response(
    mocked_chat_service: ChatProcessorService,
    mocker
):
    """Test handling invalid response structure from retrieval service."""

    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)
    mock_make_request.return_value = {"invalid_key": "some data"} # Does not match RetrievalResponse

    query = "test query"
    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service._call_retrieval_service(query)

    mock_make_request.assert_awaited_once() # Check it was awaited
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "An unexpected error occurred while retrieving context" in exc_info.value.detail


# Test _call_generation_service
async def test_call_generation_service_success(
    mocked_chat_service: ChatProcessorService,
    override_settings: Settings,
    mocker
):
    """Test successful call to the generation service."""

    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)
    expected_answer = "This is the generated answer."
    mock_response_data = GenerationResponse(answer=expected_answer).model_dump()
    mock_make_request.return_value = mock_response_data

    query = "test query"
    context = ["chunk 1", "chunk 2"]
    generated_answer = await mocked_chat_service._call_generation_service(query, context)

    expected_url = f"{str(override_settings.GENERATION_SERVICE_URL)}generate"
    expected_payload = GenerationRequest(query=query, context_chunks=context).model_dump()

    mock_make_request.assert_awaited_once_with(
        client=mocked_chat_service.http_client,
        method="POST",
        url=expected_url,
        json_data=expected_payload
    )
    assert generated_answer == expected_answer

async def test_call_generation_service_http_error(
    mocked_chat_service: ChatProcessorService,
    mocker
):
    """Test handling HTTP error from generation service."""

    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)
    original_exception = HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM Error")
    mock_make_request.side_effect = original_exception

    query = "test query"
    context = ["chunk 1"]
    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service._call_generation_service(query, context)

    mock_make_request.assert_awaited_once() # Check it was awaited
    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Generation service failed: LLM Error" in exc_info.value.detail

async def test_call_generation_service_unexpected_error(
    mocked_chat_service: ChatProcessorService,
    mocker
):
    """Test handling unexpected error during generation call."""
    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)
    mock_make_request.side_effect = ValueError("Bad input")

    query = "test query"
    context = ["chunk 1"]
    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service._call_generation_service(query, context)

    mock_make_request.assert_awaited_once() # Check it was awaited
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "unexpected error occurred while generating the response" in exc_info.value.detail

async def test_call_generation_service_invalid_response(
    mocked_chat_service: ChatProcessorService,
    mocker
):
    """Test handling invalid response structure from generation service."""

    mock_make_request = mocker.patch('app.services.chat_processor.make_request', new_callable=AsyncMock)
    mock_make_request.return_value = {"wrong_field": "no answer"} # Does not match GenerationResponse

    query = "test query"
    context = ["chunk 1"]
    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service._call_generation_service(query, context)

    mock_make_request.assert_awaited_once() # Check it was awaited
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "An unexpected error occurred while generating the response" in exc_info.value.detail


# Test process_message (Orchestration)
# These tests mock internal service methods (_call_retrieval_service, _call_generation_service)
# directly on the mocked_chat_service instance.
async def test_process_message_success(mocked_chat_service: ChatProcessorService, mocker):
    """Test the successful orchestration of retrieval and generation."""
    # Mock the internal methods directly for orchestration test
    mock_retrieval = mocker.patch.object(mocked_chat_service, '_call_retrieval_service', new_callable=AsyncMock)
    mock_generation = mocker.patch.object(mocked_chat_service, '_call_generation_service', new_callable=AsyncMock)

    retrieved_context = ["context chunk 1"]
    final_answer = "The final generated answer."
    mock_retrieval.return_value = retrieved_context
    mock_generation.return_value = final_answer

    request = ChatRequest(user_id="test_user", message="What is RAG?")
    result = await mocked_chat_service.process_message(request)

    mock_retrieval.assert_awaited_once_with(query=request.message)
    mock_generation.assert_awaited_once_with(query=request.message, context_chunks=retrieved_context)
    assert result == final_answer

async def test_process_message_no_context_found(mocked_chat_service: ChatProcessorService, mocker):
    """Test orchestration when retrieval returns no context."""
    mock_retrieval = mocker.patch.object(mocked_chat_service, '_call_retrieval_service', new_callable=AsyncMock)
    mock_generation = mocker.patch.object(mocked_chat_service, '_call_generation_service', new_callable=AsyncMock)

    retrieved_context = [] # Simulate no context found
    final_answer = "Generated answer without context."
    mock_retrieval.return_value = retrieved_context
    mock_generation.return_value = final_answer

    request = ChatRequest(user_id="test_user", message="Obscure question?")
    result = await mocked_chat_service.process_message(request)

    mock_retrieval.assert_awaited_once_with(query=request.message)
    # Generation should still be called, but with an empty context list
    mock_generation.assert_awaited_once_with(query=request.message, context_chunks=[])
    assert result == final_answer

async def test_process_message_retrieval_fails(mocked_chat_service: ChatProcessorService, mocker):
    """Test orchestration when the retrieval step fails."""
    mock_retrieval = mocker.patch.object(mocked_chat_service, '_call_retrieval_service', new_callable=AsyncMock)
    mock_generation = mocker.patch.object(mocked_chat_service, '_call_generation_service', new_callable=AsyncMock)

    retrieval_exception = HTTPException(status_code=503, detail="Retrieval Down")
    mock_retrieval.side_effect = retrieval_exception

    request = ChatRequest(user_id="test_user", message="What is RAG?")

    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service.process_message(request)

    mock_retrieval.assert_awaited_once_with(query=request.message)
    mock_generation.assert_not_awaited() # Generation should not be called
    assert exc_info.value == retrieval_exception # Exception should propagate

async def test_process_message_generation_fails(mocked_chat_service: ChatProcessorService, mocker):
    """Test orchestration when the generation step fails."""
    mock_retrieval = mocker.patch.object(mocked_chat_service, '_call_retrieval_service', new_callable=AsyncMock)
    mock_generation = mocker.patch.object(mocked_chat_service, '_call_generation_service', new_callable=AsyncMock)

    retrieved_context = ["context chunk 1"]
    generation_exception = HTTPException(status_code=503, detail="Generation Down")
    mock_retrieval.return_value = retrieved_context
    mock_generation.side_effect = generation_exception

    request = ChatRequest(user_id="test_user", message="What is RAG?")

    with pytest.raises(HTTPException) as exc_info:
        await mocked_chat_service.process_message(request)

    mock_retrieval.assert_awaited_once_with(query=request.message)
    mock_generation.assert_awaited_once_with(query=request.message, context_chunks=retrieved_context)
    assert exc_info.value == generation_exception # Exception should propagate