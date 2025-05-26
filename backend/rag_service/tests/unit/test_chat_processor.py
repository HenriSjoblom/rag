"""
Unit tests for ChatProcessorService in the RAG service.
"""

import pytest
from app.services.chat_processor import ChatProcessorService
from fastapi import HTTPException


class TestChatProcessorServiceInitialization:
    """Test cases for ChatProcessorService initialization."""

    def test_service_initialization_success(self, mock_http_client):
        """Test successful service initialization."""
        service = ChatProcessorService(
            retrieval_service_url="http://retrieval-service",
            generation_service_url="http://generation-service",
            http_client=mock_http_client,
        )

        assert service.retrieval_service_url == "http://retrieval-service"
        assert service.generation_service_url == "http://generation-service"
        assert service.http_client == mock_http_client

    def test_service_initialization_with_trailing_slashes(self, mock_http_client):
        """Test service initialization with URLs containing trailing slashes."""
        service = ChatProcessorService(
            retrieval_service_url="http://retrieval-service/",
            generation_service_url="http://generation-service/",
            http_client=mock_http_client,
        )

        assert service.retrieval_service_url == "http://retrieval-service/"
        assert service.generation_service_url == "http://generation-service/"


class TestCallRetrievalService:
    """Test cases for _call_retrieval_service method."""

    @pytest.mark.asyncio
    async def test_call_retrieval_service_success(self, chat_processor_service, mocker):
        """Test successful call to the retrieval service."""
        # Mock the make_request function
        mock_make_request = mocker.patch(
            "app.services.chat_processor.make_request",
            return_value={"chunks": ["chunk 1", "chunk 2", "chunk 3"]},
        )

        result = await chat_processor_service._call_retrieval_service("test query")

        # Verify the request was made correctly
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["url"].endswith("/api/v1/retrieve")
        assert call_args[1]["json_data"]["query"] == "test query"
        assert call_args[1]["json_data"]["top_k"] == 5

        # Verify the result structure
        assert len(result) == 3
        assert all("text" in chunk for chunk in result)
        assert result[0]["text"] == "chunk 1"
        assert result[1]["text"] == "chunk 2"
        assert result[2]["text"] == "chunk 3"

    @pytest.mark.asyncio
    async def test_call_retrieval_service_with_custom_top_k(
        self, chat_processor_service, mocker
    ):
        """Test call to retrieval service with custom top_k parameter."""
        mock_make_request = mocker.patch(
            "app.services.chat_processor.make_request",
            return_value={"chunks": ["chunk 1"]},
        )

        await chat_processor_service._call_retrieval_service("test query", top_k=10)

        call_args = mock_make_request.call_args
        assert call_args[1]["json_data"]["top_k"] == 10

    @pytest.mark.asyncio
    async def test_call_retrieval_service_empty_chunks(
        self, chat_processor_service, mocker
    ):
        """Test retrieval service returning empty chunks."""
        mocker.patch(
            "app.services.chat_processor.make_request", return_value={"chunks": []}
        )

        result = await chat_processor_service._call_retrieval_service("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_call_retrieval_service_http_exception(
        self, chat_processor_service, mocker
    ):
        """Test handling HTTPException from retrieval service."""
        mocker.patch(
            "app.services.chat_processor.make_request",
            side_effect=HTTPException(status_code=404, detail="Not found"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service._call_retrieval_service("test query")

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"

    @pytest.mark.asyncio
    async def test_call_retrieval_service_validation_error(
        self, chat_processor_service, mocker
    ):
        """Test handling validation error from retrieval service response."""
        mocker.patch(
            "app.services.chat_processor.make_request",
            return_value={"invalid": "response"},
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service._call_retrieval_service("test query")

        assert exc_info.value.status_code == 500
        assert "failed validation" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_call_retrieval_service_unexpected_error(
        self, chat_processor_service, mocker
    ):
        """Test handling unexpected errors from retrieval service."""
        mocker.patch(
            "app.services.chat_processor.make_request",
            side_effect=Exception("Connection failed"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service._call_retrieval_service("test query")

        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.detail


class TestCallGenerationService:
    """Test cases for _call_generation_service method."""

    @pytest.mark.asyncio
    async def test_call_generation_service_success(
        self, chat_processor_service, mocker
    ):
        """Test successful call to the generation service."""
        mock_make_request = mocker.patch(
            "app.services.chat_processor.make_request",
            return_value={"answer": "This is the generated response."},
        )

        context_chunks = [{"text": "chunk 1"}, {"text": "chunk 2"}]

        result = await chat_processor_service._call_generation_service(
            "test query", context_chunks
        )

        # Verify the request was made correctly
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["url"].endswith("/api/v1/generate")
        assert call_args[1]["json_data"]["query"] == "test query"
        assert call_args[1]["json_data"]["context_chunks"] == ["chunk 1", "chunk 2"]

        # Verify the result
        assert result == "This is the generated response."

    @pytest.mark.asyncio
    async def test_call_generation_service_empty_context(
        self, chat_processor_service, mocker
    ):
        """Test generation service call with empty context chunks."""
        mock_make_request = mocker.patch(
            "app.services.chat_processor.make_request",
            return_value={"answer": "Response without context."},
        )

        result = await chat_processor_service._call_generation_service("test query", [])

        call_args = mock_make_request.call_args
        assert call_args[1]["json_data"]["context_chunks"] == []
        assert result == "Response without context."

    @pytest.mark.asyncio
    async def test_call_generation_service_filters_none_text(
        self, chat_processor_service, mocker
    ):
        """Test generation service filters out chunks with None text."""
        mock_make_request = mocker.patch(
            "app.services.chat_processor.make_request",
            return_value={"answer": "Filtered response."},
        )

        context_chunks = [
            {"text": "chunk 1"},
            {"text": None},
            {"text": "chunk 2"},
            {"other": "data"},  # No text key
        ]

        await chat_processor_service._call_generation_service(
            "test query", context_chunks
        )

        call_args = mock_make_request.call_args
        # Should only include chunks with valid text (None text is filtered out)
        assert call_args[1]["json_data"]["context_chunks"] == ["chunk 1", "chunk 2"]

    @pytest.mark.asyncio
    async def test_call_generation_service_http_exception(
        self, chat_processor_service, mocker
    ):
        """Test handling HTTPException from generation service."""
        mocker.patch(
            "app.services.chat_processor.make_request",
            side_effect=HTTPException(status_code=500, detail="Internal server error"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service._call_generation_service("test query", [])

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error"

    @pytest.mark.asyncio
    async def test_call_generation_service_unexpected_error(
        self, chat_processor_service, mocker
    ):
        """Test handling unexpected errors from generation service."""
        mocker.patch(
            "app.services.chat_processor.make_request",
            side_effect=Exception("Service unavailable"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service._call_generation_service("test query", [])

        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.detail


class TestProcessMethod:
    """Test cases for the main process method."""

    @pytest.mark.asyncio
    async def test_process_success(self, chat_processor_service, mocker):
        """Test successful end-to-end processing."""
        # Mock the private methods
        mock_retrieval = mocker.patch.object(
            chat_processor_service,
            "_call_retrieval_service",
            return_value=[{"text": "retrieved chunk"}],
        )
        mock_generation = mocker.patch.object(
            chat_processor_service,
            "_call_generation_service",
            return_value="Generated response",
        )

        result = await chat_processor_service.process("user123", "test query")

        # Verify calls
        mock_retrieval.assert_called_once_with(query="test query")
        mock_generation.assert_called_once_with(
            user_query="test query", context_chunks=[{"text": "retrieved chunk"}]
        )

        assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_process_empty_retrieval_results(
        self, chat_processor_service, mocker
    ):
        """Test processing with empty retrieval results."""
        mock_retrieval = mocker.patch.object(
            chat_processor_service, "_call_retrieval_service", return_value=[]
        )
        mock_generation = mocker.patch.object(
            chat_processor_service,
            "_call_generation_service",
            return_value="Response without context",
        )

        result = await chat_processor_service.process("user123", "test query")

        mock_retrieval.assert_called_once_with(query="test query")
        mock_generation.assert_called_once_with(
            user_query="test query", context_chunks=[]
        )

        assert result == "Response without context"

    @pytest.mark.asyncio
    async def test_process_retrieval_failure(self, chat_processor_service, mocker):
        """Test processing when retrieval service fails."""
        mocker.patch.object(
            chat_processor_service,
            "_call_retrieval_service",
            side_effect=HTTPException(
                status_code=503, detail="Retrieval service unavailable"
            ),
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service.process("user123", "test query")

        assert exc_info.value.status_code == 503
        assert "Error from retrieval" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_generation_failure(self, chat_processor_service, mocker):
        """Test processing when generation service fails."""
        mocker.patch.object(
            chat_processor_service,
            "_call_retrieval_service",
            return_value=[{"text": "context"}],
        )
        mocker.patch.object(
            chat_processor_service,
            "_call_generation_service",
            side_effect=HTTPException(
                status_code=500, detail="Generation service error"
            ),
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service.process("user123", "test query")

        assert exc_info.value.status_code == 500
        assert "Error from generation" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_unexpected_generation_error(
        self, chat_processor_service, mocker
    ):
        """Test processing with unexpected error during generation."""
        mocker.patch.object(
            chat_processor_service,
            "_call_retrieval_service",
            return_value=[{"text": "context"}],
        )
        mocker.patch.object(
            chat_processor_service,
            "_call_generation_service",
            side_effect=Exception("Unexpected error"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_processor_service.process("user123", "test query")

        assert exc_info.value.status_code == 500
        assert "unexpected error" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_logs_user_info(self, chat_processor_service, mocker, caplog):
        """Test that process method logs user information."""
        mocker.patch.object(
            chat_processor_service,
            "_call_retrieval_service",
            return_value=[{"text": "context"}],
        )
        mocker.patch.object(
            chat_processor_service, "_call_generation_service", return_value="Response"
        )

        await chat_processor_service.process("user123", "test query")

        # Check that user ID is logged
        assert "user_id: user123" in caplog.text
        assert "Generated AI response for user_id: user123" in caplog.text
