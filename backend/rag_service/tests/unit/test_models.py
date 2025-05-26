"""
Unit tests for Pydantic models in the RAG service.
"""

import pytest
from app.models import (
    ChatRequest,
    ChatResponse,
    GenerationRequest,
    GenerationResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from pydantic import ValidationError


class TestChatRequest:
    """Test cases for ChatRequest model."""

    def test_chat_request_valid(self):
        """Test ChatRequest with valid data."""
        request = ChatRequest(message="Hello, how are you?")
        assert request.message == "Hello, how are you?"

    def test_chat_request_empty_message(self):
        """Test that empty message raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_chat_request_missing_user_id(self):
        """Test that missing message raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest()  # missing message
        assert "Field required" in str(exc_info.value)

    def test_chat_request_missing_message(self):
        """Test that missing message raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest()  # missing message
        assert "Field required" in str(exc_info.value)

    def test_chat_request_whitespace_message(self):
        """Test ChatRequest with whitespace message."""
        request = ChatRequest(message="   Hello   ")
        assert request.message == "   Hello   "

    def test_chat_request_unicode(self):
        """Test ChatRequest with unicode characters."""
        request = ChatRequest(message="Hello with émojis 🤖")
        assert request.message == "Hello with émojis 🤖"

    def test_chat_request_very_long_message(self):
        """Test ChatRequest with very long message."""
        long_message = "Hello " * 1000
        request = ChatRequest(message=long_message)
        assert request.message == long_message


class TestChatResponse:
    """Test cases for ChatResponse model."""

    def test_chat_response_valid(self):
        """Test ChatResponse with valid data."""
        response = ChatResponse(query="How are you?", response="Hello, I'm doing well!")
        assert response.response == "Hello, I'm doing well!"

    def test_chat_response_empty_response(self):
        """Test ChatResponse with empty response."""
        response = ChatResponse(query="How are you?", response="")
        assert response.response == ""

    def test_chat_response_unicode(self):
        """Test ChatResponse with unicode characters."""
        response = ChatResponse(
            query="How are you?",
            response="Response with émojis 🤖",
        )
        assert response.response == "Response with émojis 🤖"

    def test_chat_response_missing_response(self):
        """Test that missing response raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatResponse(query="test")  # missing response
        assert "Field required" in str(exc_info.value)


class TestRetrievalRequest:
    """Test cases for RetrievalRequest model."""

    def test_retrieval_request_valid(self):
        """Test RetrievalRequest with valid query."""
        request = RetrievalRequest(query="test query")
        assert request.query == "test query"

    def test_retrieval_request_empty_query(self):
        """Test RetrievalRequest with empty query."""
        request = RetrievalRequest(query="")
        assert request.query == ""

    def test_retrieval_request_missing_query(self):
        """Test that missing query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RetrievalRequest()  # missing query
        assert "Field required" in str(exc_info.value)

    def test_retrieval_request_unicode(self):
        """Test RetrievalRequest with unicode characters."""
        request = RetrievalRequest(query="unicode 🤖 test")
        assert request.query == "unicode 🤖 test"


class TestRetrievalResponse:
    """Test cases for RetrievalResponse model."""

    def test_retrieval_response_valid(self):
        """Test RetrievalResponse with valid chunks."""
        response = RetrievalResponse(chunks=["chunk1", "chunk2"])
        assert response.chunks == ["chunk1", "chunk2"]

    def test_retrieval_response_empty_chunks(self):
        """Test RetrievalResponse with empty chunks."""
        response = RetrievalResponse(chunks=[])
        assert response.chunks == []

    def test_retrieval_response_missing_chunks(self):
        """Test that missing chunks raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RetrievalResponse()  # missing chunks
        assert "Field required" in str(exc_info.value)

    def test_retrieval_response_validation(self):
        """Test RetrievalResponse validation with invalid chunks."""
        with pytest.raises(ValidationError):
            RetrievalResponse(chunks="not a list")

    def test_retrieval_response_unicode_chunks(self):
        """Test RetrievalResponse with unicode chunks."""
        response = RetrievalResponse(chunks=["chunk 🤖", "test"])
        assert response.chunks == ["chunk 🤖", "test"]


class TestGenerationRequest:
    """Test cases for GenerationRequest model."""

    def test_generation_request_valid(self):
        """Test GenerationRequest with valid data."""
        request = GenerationRequest(query="test query", context_chunks=["chunk1"])
        assert request.query == "test query"
        assert request.context_chunks == ["chunk1"]

    def test_generation_request_empty_context(self):
        """Test GenerationRequest with empty context."""
        request = GenerationRequest(query="test query", context_chunks=[])
        assert request.context_chunks == []

    def test_generation_request_missing_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(query="test")  # missing context_chunks
        assert "Field required" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(context_chunks=["chunk"])  # missing query
        assert "Field required" in str(exc_info.value)

    def test_generation_request_validation(self):
        """Test GenerationRequest validation with invalid context_chunks."""
        with pytest.raises(ValidationError):
            GenerationRequest(query="test", context_chunks="not a list")

    def test_generation_request_unicode(self):
        """Test GenerationRequest with unicode characters."""
        request = GenerationRequest(
            query="unicode 🤖 query", context_chunks=["context 🤖"]
        )
        assert request.query == "unicode 🤖 query"
        assert request.context_chunks == ["context 🤖"]


class TestGenerationResponse:
    """Test cases for GenerationResponse model."""

    def test_generation_response_valid(self):
        """Test GenerationResponse with valid answer."""
        response = GenerationResponse(answer="test answer")
        assert response.answer == "test answer"

    def test_generation_response_empty_answer(self):
        """Test GenerationResponse with empty answer."""
        response = GenerationResponse(answer="")
        assert response.answer == ""

    def test_generation_response_missing_answer(self):
        """Test that missing answer raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationResponse()  # missing answer
        assert "Field required" in str(exc_info.value)

    def test_generation_response_unicode(self):
        """Test GenerationResponse with unicode characters."""
        response = GenerationResponse(answer="unicode 🤖 answer")
        assert response.answer == "unicode 🤖 answer"
