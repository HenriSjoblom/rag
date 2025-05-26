"""
Unit tests for Pydantic models in the generation service.
"""

import pytest
from app.models import GenerateRequest, GenerateResponse
from pydantic import ValidationError


class TestGenerateRequest:
    """Test cases for GenerateRequest model."""

    def test_generate_request_valid_data(self):
        """Test GenerateRequest with valid data."""
        request = GenerateRequest(
            query="What is FastAPI?", context_chunks=["FastAPI is a web framework."]
        )
        assert request.query == "What is FastAPI?"
        assert request.context_chunks == ["FastAPI is a web framework."]

    def test_generate_request_empty_query(self):
        """Test that empty query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(query="", context_chunks=["Some context"])
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_generate_request_empty_context_chunks(self):
        """Test GenerateRequest with empty context chunks."""
        request = GenerateRequest(query="What is FastAPI?", context_chunks=[])
        assert request.query == "What is FastAPI?"
        assert request.context_chunks == []

    def test_generate_request_none_context_chunks(self):
        """Test GenerateRequest with None context chunks."""
        with pytest.raises(ValidationError):
            GenerateRequest(query="What is FastAPI?", context_chunks=None)

    def test_generate_request_serialization(self):
        """Test that GenerateRequest can be serialized to dict."""
        request = GenerateRequest(
            query="Test query", context_chunks=["Context 1", "Context 2"]
        )
        data = request.model_dump()
        expected = {"query": "Test query", "context_chunks": ["Context 1", "Context 2"]}
        assert data == expected


class TestGenerateResponse:
    """Test cases for GenerateResponse model."""

    def test_generate_response_valid_data(self):
        """Test GenerateResponse with valid data."""
        response = GenerateResponse(answer="This is the answer.")
        assert response.answer == "This is the answer."

    def test_generate_response_empty_answer(self):
        """Test that empty answer is allowed (no validation constraint on answer length)."""
        response = GenerateResponse(answer="")
        assert response.answer == ""

    def test_generate_response_long_answer(self):
        """Test GenerateResponse with long answer."""
        long_answer = "This is a very long answer. " * 100
        response = GenerateResponse(answer=long_answer)
        assert response.answer == long_answer

    def test_generate_response_serialization(self):
        """Test that GenerateResponse can be serialized to dict."""
        response = GenerateResponse(answer="Test answer")
        data = response.model_dump()
        expected = {"answer": "Test answer"}
        assert data == expected

    def test_generate_response_with_special_characters(self):
        """Test GenerateResponse with special characters."""
        special_answer = "Answer with émojis 🚀 and special chars: @#$%^&*()"
        response = GenerateResponse(answer=special_answer)
        assert response.answer == special_answer
