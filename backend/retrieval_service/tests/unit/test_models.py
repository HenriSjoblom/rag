"""
Unit tests for Pydantic models in the retrieval service.
"""

import pytest
from app.models import RetrievalRequest, RetrievalResponse
from pydantic import ValidationError


class TestRetrievalRequest:
    """Test cases for RetrievalRequest model."""

    def test_retrieval_request_valid(self):
        """Test RetrievalRequest with valid query."""
        request = RetrievalRequest(query="test query")
        assert request.query == "test query"

    def test_retrieval_request_empty_query(self):
        """Test that empty query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RetrievalRequest(query="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_retrieval_request_whitespace_query(self):
        """Test RetrievalRequest with whitespace query."""
        request = RetrievalRequest(query="   test query   ")
        assert request.query == "   test query   "

    def test_retrieval_request_unicode(self):
        """Test RetrievalRequest with unicode characters."""
        request = RetrievalRequest(query="test query with émojis 🔍")
        assert request.query == "test query with émojis 🔍"

    def test_retrieval_request_very_long(self):
        """Test RetrievalRequest with very long query."""
        long_query = "test " * 1000
        request = RetrievalRequest(query=long_query)
        assert request.query == long_query


class TestRetrievalResponse:
    """Test cases for RetrievalResponse model."""

    def test_retrieval_response_minimal(self):
        """Test RetrievalResponse with minimal data."""
        response = RetrievalResponse()
        assert response.chunks == []
        assert response.collection_name is None
        assert response.query is None

    def test_retrieval_response_full(self):
        """Test RetrievalResponse with all fields."""
        response = RetrievalResponse(
            chunks=["chunk1", "chunk2"],
            collection_name="test_collection",
            query="test query",
        )
        assert response.chunks == ["chunk1", "chunk2"]
        assert response.collection_name == "test_collection"
        assert response.query == "test query"

    def test_retrieval_response_empty_chunks(self):
        """Test RetrievalResponse with empty chunks list."""
        response = RetrievalResponse(
            chunks=[],
            collection_name="test_collection",
            query="no results query",
        )
        assert response.chunks == []
        assert response.collection_name == "test_collection"
        assert response.query == "no results query"

    def test_retrieval_response_many_chunks(self):
        """Test RetrievalResponse with many chunks."""
        chunks = [f"chunk_{i}" for i in range(100)]
        response = RetrievalResponse(
            chunks=chunks,
            collection_name="test_collection",
            query="test query",
        )
        assert len(response.chunks) == 100
        assert response.chunks[0] == "chunk_0"
        assert response.chunks[-1] == "chunk_99"

    def test_retrieval_response_validation(self):
        """Test RetrievalResponse field validation."""
        # chunks must be a list
        with pytest.raises(ValidationError):
            RetrievalResponse(chunks="not a list")

        # collection_name can be None or string
        response1 = RetrievalResponse(collection_name=None)
        assert response1.collection_name is None

        response2 = RetrievalResponse(collection_name="test_collection")
        assert response2.collection_name == "test_collection"

        # query can be None or string
        response3 = RetrievalResponse(query=None)
        assert response3.query is None

        response4 = RetrievalResponse(query="test query")
        assert response4.query == "test query"
