import pytest
from unittest.mock import AsyncMock, MagicMock, ANY

from fastapi import status, HTTPException
from fastapi.testclient import TestClient

from app.models import ChatRequest, ChatResponse

from app.config import Settings
from app.services.chat_processor import ChatProcessorService
from app.deps import get_chat_processor_service # Used for potential overrides if needed

pytestmark = pytest.mark.usefixtures("client")

# --- Integration Tests for RAG Service API ---

def test_handle_chat_message_success(
    client: TestClient,
    mocker,
):
    """Test successful chat message processing."""
    mocked_process_message = mocker.patch(
        'app.services.chat_processor.ChatProcessorService.process_message',
        new_callable=AsyncMock  # For async methods
    )

    expected_bot_response = "This is a helpful RAG response."
    mocked_process_message.return_value = expected_bot_response

    payload = {"user_id": "test_user_success", "message": "Hello RAG!"}

    response = client.post("/api/v1/chat", json=payload) # Endpoint from rag_service/app/routers.py

    assert response.status_code == status.HTTP_200_OK, \
        f"Expected 200, got {response.status_code}. Response: {response.text}"

    data = response.json()
    assert data["bot_response"] == expected_bot_response

    expected_request_obj = ChatRequest(user_id="test_user_success", message="Hello RAG!")
    # The first argument to the mocked method will be 'self' (the instance)
    #mocked_process_message.assert_awaited_once_with(ANY, expected_request_obj)
    mocked_process_message.assert_awaited_once()
     # Access arguments using .args and .kwargs from the call_args object
    actual_pos_args = mocked_process_message.call_args.args
    actual_kwargs = mocked_process_message.call_args.kwargs

    print(f"DEBUG: Expected request: {expected_request_obj.model_dump_json(indent=2)}")
    print(f"DEBUG: Actual positional args: {actual_pos_args}")
    print(f"DEBUG: Actual keyword args: {actual_kwargs}")

    # Assuming process_message is called as: await self.process_message(request_obj)
    # The 'self' instance will be the first positional argument.
    # The ChatRequest object will be the second positional argument.
    assert len(actual_pos_args) == 1, "Expected one positional argument (the request object)"

   # The first argument in actual_pos_args is our ChatRequest object
    actual_request_obj = actual_pos_args[0]

    print(f"DEBUG: Actual request type: {type(actual_request_obj)}")
    if isinstance(actual_request_obj, ChatRequest):
         print(f"DEBUG: Actual request: {actual_request_obj.model_dump_json(indent=2)}")
    else:
        print(f"DEBUG: Actual request (not ChatRequest): {actual_request_obj}")

    assert actual_request_obj == expected_request_obj # Pydantic models have __eq__

    # If there were keyword arguments you expected, you would assert them here:
    assert not actual_kwargs, "No keyword arguments were expected"

def test_handle_chat_message_service_http_exception(
    client: TestClient,
    mocker,
):
    """Test handling of HTTPException raised by the chat service."""
    mocked_process_message = mocker.patch(
        'app.services.chat_processor.ChatProcessorService.process_message',
        new_callable=AsyncMock
    )

    service_exception = HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="RAG service is temporarily down."
    )
    mocked_process_message.side_effect = service_exception

    payload = {"user_id": "test_user_http_error", "message": "Query during service outage"}

    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    data = response.json()
    assert data["detail"] == "RAG service is temporarily down."

    expected_request_obj = ChatRequest(user_id="test_user_http_error", message="Query during service outage")
    mocked_process_message.assert_awaited_once_with(expected_request_obj)

def test_handle_chat_message_service_unexpected_exception(
    client: TestClient,
    mocker,
):
    """Test handling of an unexpected Exception raised by the chat service."""
    mocked_process_message = mocker.patch(
        'app.services.chat_processor.ChatProcessorService.process_message',
        new_callable=AsyncMock
    )

    service_exception = ValueError("Something went very wrong internally.")
    mocked_process_message.side_effect = service_exception

    payload = {"user_id": "test_user_unexpected_error", "message": "Query causing an internal error"}

    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    data = response.json()

    assert data["detail"] == "An internal error occurred while processing your message."

    expected_request_obj = ChatRequest(user_id="test_user_unexpected_error", message="Query causing an internal error")
    mocked_process_message.assert_awaited_once_with(expected_request_obj)

def test_handle_chat_message_invalid_request_payload_missing_field(
    client: TestClient,
):
    """Test chat endpoint with a request payload missing a required field."""
    invalid_payload = {"user_id": "test_user_invalid_payload"}  # 'message' field is missing

    response = client.post("/api/v1/chat", json=invalid_payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    # For debugging, you can print the exact error details:
    # print("DEBUG: Validation error details:", data.get("detail"))
    assert "detail" in data

    found_error = any(
        # Check if 'message' is the field causing the error.
        # Pydantic v2 loc for body fields is typically ['body', 'field_name']
        # This check is somewhat lenient and should still work if loc is ['body', 'message']
        "message" in error.get("loc", []) and
        # Correctly check for Pydantic v2's 'missing' type (lowercase)
        # or fall back to Pydantic v1's 'value_error.missing'
        (error.get("type") == "missing" or error.get("type") == "value_error.missing")
        for error in data.get("detail", [])
    )
    assert found_error, "Validation error for missing 'message' field not found in response."

def test_handle_chat_message_invalid_request_payload_wrong_type(
    client: TestClient,
):
    """Test chat endpoint with a request payload having a field with the wrong type."""
    invalid_payload = {"user_id": 12345, "message": "Valid message content"}  # 'user_id' should be a string

    response = client.post("/api/v1/chat", json=invalid_payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert "detail" in data
    # Check for a message indicating 'user_id' field has a type error
    found_error = any(
        "user_id" in error.get("loc", []) and
        ("string_type" in error.get("type", "") or "str_type" in error.get("type") or "type_error.str" == error.get("type"))
        for error in data.get("detail", [])
    )
    assert found_error, "Validation error for 'user_id' field type not found in response."


def test_health_check(client: TestClient):
     """Test the health check endpoint if it exists."""
     response = client.get("/health") # Assuming a /health endpoint
     assert response.status_code == status.HTTP_200_OK
     assert response.json().get("status") == "ok"