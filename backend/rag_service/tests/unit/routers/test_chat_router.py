"""
Unit tests for chat router endpoints in the RAG service.
"""

import logging

import pytest
from app.models import ChatRequest, ChatResponse
from app.routers.chat import process_chat_message
from fastapi import HTTPException


class TestProcessChatMessage:
    """Test cases for process_chat_message endpoint."""

    @pytest.mark.asyncio
    async def test_process_chat_message_success(
        self, mock_chat_processor_service, mocker
    ):
        """Test successful chat message processing."""
        # Setup
        mock_chat_processor_service.process = mocker.AsyncMock(
            return_value="This is the AI response."
        )
        chat_request = ChatRequest(message="Hello, how are you?")

        # Execute
        result = await process_chat_message(
            chat_request=chat_request, chat_processor=mock_chat_processor_service
        )

        # Verify
        mock_chat_processor_service.process.assert_called_once_with(
            query="Hello, how are you?"
        )

        assert isinstance(result, ChatResponse)
        assert result.query == "Hello, how are you?"
        assert result.response == "This is the AI response."

    @pytest.mark.asyncio
    async def test_process_chat_message_empty_message(
        self, mock_chat_processor_service, mocker
    ):
        """Test chat processing with minimal message."""
        # This test verifies that Pydantic validation passes for minimal valid message
        mock_chat_processor_service.process = mocker.AsyncMock(return_value="Response")

        chat_request = ChatRequest(message="a")  # minimum valid message
        result = await process_chat_message(
            chat_request=chat_request, chat_processor=mock_chat_processor_service
        )

        assert isinstance(result, ChatResponse)
        assert result.query == "a"

    @pytest.mark.asyncio
    async def test_process_chat_message_long_message(
        self, mock_chat_processor_service, mocker
    ):
        """Test chat processing with very long message."""
        mock_chat_processor_service.process = mocker.AsyncMock(
            return_value="Response to long message"
        )
        long_message = "This is a very long message. " * 100

        chat_request = ChatRequest(message=long_message)

        result = await process_chat_message(
            chat_request=chat_request, chat_processor=mock_chat_processor_service
        )

        mock_chat_processor_service.process.assert_called_once_with(query=long_message)
        assert result.response == "Response to long message"

    @pytest.mark.asyncio
    async def test_process_chat_message_unicode_characters(
        self, mock_chat_processor_service, mocker
    ):
        """Test chat processing with unicode characters."""
        mock_chat_processor_service.process = mocker.AsyncMock(
            return_value="Response with unicode 🤖"
        )
        chat_request = ChatRequest(message="Hello with émojis 🤖")

        result = await process_chat_message(
            chat_request=chat_request, chat_processor=mock_chat_processor_service
        )

        assert result.query == "Hello with émojis 🤖"
        assert result.response == "Response with unicode 🤖"

    @pytest.mark.asyncio
    async def test_process_chat_message_http_exception_503(
        self, mock_chat_processor_service, mocker
    ):
        """Test handling of HTTPException from chat processor (service unavailable)."""
        mock_chat_processor_service.process = mocker.AsyncMock(
            side_effect=HTTPException(
                status_code=503, detail="Retrieval service is unavailable"
            )
        )

        chat_request = ChatRequest(message="Test message")

        with pytest.raises(HTTPException) as exc_info:
            await process_chat_message(
                chat_request=chat_request, chat_processor=mock_chat_processor_service
            )

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Retrieval service is unavailable"

    @pytest.mark.asyncio
    async def test_process_chat_message_http_exception_500(
        self, mock_chat_processor_service, mocker
    ):
        """Test handling of HTTPException from chat processor (internal server error)."""
        mock_chat_processor_service.process = mocker.AsyncMock(
            side_effect=HTTPException(status_code=500, detail="Internal server error")
        )

        chat_request = ChatRequest(message="Test message")

        with pytest.raises(HTTPException) as exc_info:
            await process_chat_message(
                chat_request=chat_request, chat_processor=mock_chat_processor_service
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error"

    @pytest.mark.asyncio
    async def test_process_chat_message_unexpected_exception(
        self, mock_chat_processor_service, mocker
    ):
        """Test handling of unexpected exceptions from chat processor."""
        mock_chat_processor_service.process = mocker.AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        chat_request = ChatRequest(message="Test message")

        with pytest.raises(HTTPException) as exc_info:
            await process_chat_message(
                chat_request=chat_request, chat_processor=mock_chat_processor_service
            )

        assert exc_info.value.status_code == 500
        assert "An unexpected error occurred: Unexpected error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_chat_message_different_user_ids(
        self, mock_chat_processor_service, mocker
    ):
        """Test processing messages (user_id field no longer exists)."""
        mock_chat_processor_service.process = mocker.AsyncMock(return_value="Response")

        # Test that the endpoint works without user_id
        chat_request = ChatRequest(message="Test message")
        result = await process_chat_message(
            chat_request=chat_request, chat_processor=mock_chat_processor_service
        )

        assert result.query == "Test message"
        assert result.response == "Response"

    @pytest.mark.asyncio
    async def test_process_chat_message_logs_truncated_message(
        self, mock_chat_processor_service, caplog, mocker
    ):
        """Test that long messages are properly truncated in logs."""
        # Set the logging level to ensure we capture INFO logs
        caplog.set_level(logging.INFO)
        mock_chat_processor_service.process = mocker.AsyncMock(return_value="Response")
        long_message = "A" * 200  # Message longer than 50 characters
        chat_request = ChatRequest(message=long_message)

        await process_chat_message(
            chat_request=chat_request, chat_processor=mock_chat_processor_service
        )

        # Check that the log message contains truncated version
        log_messages = [record.message for record in caplog.records]
        truncated_log = next(
            (
                msg
                for msg in log_messages
                if "Received chat request, message:" in msg and "..." in msg
            ),
            None,
        )
        assert truncated_log is not None

    @pytest.mark.asyncio
    async def test_process_chat_message_logs_success(
        self, mock_chat_processor_service, caplog, mocker
    ):
        """Test that successful processing is logged."""
        # Set the logging level to ensure we capture INFO logs
        caplog.set_level(logging.INFO)
        mock_chat_processor_service.process = mocker.AsyncMock(
            return_value="Success response"
        )
        chat_request = ChatRequest(message="Test message")

        await process_chat_message(
            chat_request=chat_request, chat_processor=mock_chat_processor_service
        )

        # Check for success log
        log_messages = [record.message for record in caplog.records]
        success_log = next(
            (
                msg
                for msg in log_messages
                if "Successfully processed chat request" in msg
            ),
            None,
        )
        assert success_log is not None

    @pytest.mark.asyncio
    async def test_process_chat_message_logs_http_exception(
        self, mock_chat_processor_service, caplog, mocker
    ):
        """Test that HTTP exceptions are properly logged."""
        mock_chat_processor_service.process = mocker.AsyncMock(
            side_effect=HTTPException(status_code=503, detail="Service unavailable")
        )
        chat_request = ChatRequest(message="Test message")

        with pytest.raises(HTTPException):
            await process_chat_message(
                chat_request=chat_request, chat_processor=mock_chat_processor_service
            )

        # Check for error log
        log_messages = [record.message for record in caplog.records]
        error_log = next(
            (
                msg
                for msg in log_messages
                if "HTTPException during chat processing" in msg
            ),
            None,
        )
        assert error_log is not None
        assert "Service unavailable" in error_log

    @pytest.mark.asyncio
    async def test_process_chat_message_logs_unexpected_exception(
        self, mock_chat_processor_service, caplog, mocker
    ):
        """Test that unexpected exceptions are properly logged."""
        mock_chat_processor_service.process = mocker.AsyncMock(
            side_effect=Exception("Unexpected error")
        )
        chat_request = ChatRequest(message="Test message")

        with pytest.raises(HTTPException):
            await process_chat_message(
                chat_request=chat_request, chat_processor=mock_chat_processor_service
            )

        # Check for error log
        log_messages = [record.message for record in caplog.records]
        error_log = next(
            (msg for msg in log_messages if "Unexpected error processing chat" in msg),
            None,
        )
        assert error_log is not None
        assert "Unexpected error" in error_log
