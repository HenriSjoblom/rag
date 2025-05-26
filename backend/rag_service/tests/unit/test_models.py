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
        request = ChatRequest(user_id="user123", message="Hello, how are you?")
        assert request.user_id == "user123"
        assert request.message == "Hello, how are you?"

    def test_chat_request_empty_message(self):
        """Test that empty message raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(user_id="user123", message="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_chat_request_missing_user_id(self):
        """Test that missing user_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="Hello")
        assert "Field required" in str(exc_info.value)

    def test_chat_request_missing_message(self):
        """Test that missing message raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(user_id="user123")
        assert "Field required" in str(exc_info.value)

    def test_chat_request_whitespace_message(self):
        """Test ChatRequest with whitespace message."""
        request = ChatRequest(user_id="user123", message="   Hello   ")
        assert request.message == "   Hello   "

    def test_chat_request_unicode(self):
        """Test ChatRequest with unicode characters."""
        request = ChatRequest(user_id="user123", message="Hello with émojis 🤖")
        assert request.message == "Hello with émojis 🤖"

    def test_chat_request_very_long_message(self):
        """Test ChatRequest with very long message."""
        long_message = "Hello " * 1000
        request = ChatRequest(user_id="user123", message=long_message)
        assert request.message == long_message


class TestChatResponse:
    """Test cases for ChatResponse model."""

    def test_chat_response_valid(self):
        """Test ChatResponse with valid data."""
        response = ChatResponse(
            user_id="test_user", query="How are you?", response="Hello, I'm doing well!"
        )
        assert response.response == "Hello, I'm doing well!"

    def test_chat_response_empty_response(self):
        """Test ChatResponse with empty response."""
        response = ChatResponse(user_id="test_user", query="How are you?", response="")
        assert response.response == ""

    def test_chat_response_unicode(self):
        """Test ChatResponse with unicode characters."""
        response = ChatResponse(
            user_id="test_user",
            query="How are you?",
            response="Response with émojis 🤖",
        )
        assert response.response == "Response with émojis 🤖"

    def test_chat_response_missing_response(self):
        """Test that missing response raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatResponse(user_id="test", query="test")  # missing response
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
            RetrievalRequest()
        assert "Field required" in str(exc_info.value)

    def test_retrieval_request_unicode(self):
        """Test RetrievalRequest with unicode characters."""
        request = RetrievalRequest(query="test query with émojis 🔍")
        assert request.query == "test query with émojis 🔍"


class TestRetrievalResponse:
    """Test cases for RetrievalResponse model."""

    def test_retrieval_response_valid(self):
        """Test RetrievalResponse with valid chunks."""
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        response = RetrievalResponse(chunks=chunks)
        assert response.chunks == chunks

    def test_retrieval_response_empty_chunks(self):
        """Test RetrievalResponse with empty chunks list."""
        response = RetrievalResponse(chunks=[])
        assert response.chunks == []

    def test_retrieval_response_missing_chunks(self):
        """Test that missing chunks raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RetrievalResponse()
        assert "Field required" in str(exc_info.value)

    def test_retrieval_response_validation(self):
        """Test RetrievalResponse field validation."""
        # chunks must be a list
        with pytest.raises(ValidationError):
            RetrievalResponse(chunks="not a list")

    def test_retrieval_response_unicode_chunks(self):
        """Test RetrievalResponse with unicode chunks."""
        chunks = ["chunk with émojis 🔍", "another chunk 🤖"]
        response = RetrievalResponse(chunks=chunks)
        assert response.chunks == chunks


class TestGenerationRequest:
    """Test cases for GenerationRequest model."""

    def test_generation_request_valid(self):
        """Test GenerationRequest with valid data."""
        request = GenerationRequest(
            query="What is AI?",
            context_chunks=[
                "AI is artificial intelligence",
                "AI can help solve problems",
            ],
        )
        assert request.query == "What is AI?"
        assert len(request.context_chunks) == 2

    def test_generation_request_empty_context(self):
        """Test GenerationRequest with empty context chunks."""
        request = GenerationRequest(query="What is AI?", context_chunks=[])
        assert request.query == "What is AI?"
        assert request.context_chunks == []

    def test_generation_request_missing_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(query="What is AI?")
        assert "Field required" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GenerationRequest(context_chunks=["context"])
        assert "Field required" in str(exc_info.value)

    def test_generation_request_validation(self):
        """Test GenerationRequest field validation."""
        # context_chunks must be a list
        with pytest.raises(ValidationError):
            GenerationRequest(query="What is AI?", context_chunks="not a list")

    def test_generation_request_unicode(self):
        """Test GenerationRequest with unicode characters."""
        request = GenerationRequest(
            query="What is AI? 🤖", context_chunks=["AI context with émojis 🔍"]
        )
        assert request.query == "What is AI? 🤖"
        assert request.context_chunks == ["AI context with émojis 🔍"]


class TestGenerationResponse:
    """Test cases for GenerationResponse model."""

    def test_generation_response_valid(self):
        """Test GenerationResponse with valid answer."""
        response = GenerationResponse(answer="AI is artificial intelligence.")
        assert response.answer == "AI is artificial intelligence."

    def test_generation_response_empty_answer(self):
        """Test GenerationResponse with empty answer."""
        response = GenerationResponse(answer="")
        assert response.answer == ""

    def test_generation_response_missing_answer(self):
        """Test that missing answer raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GenerationResponse()
        assert "Field required" in str(exc_info.value)

    def test_generation_response_unicode(self):
        """Test GenerationResponse with unicode characters."""
        response = GenerationResponse(answer="AI response with émojis 🤖")
        assert response.answer == "AI response with émojis 🤖"
