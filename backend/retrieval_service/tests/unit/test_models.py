"""
Unit tests for Pydantic models in the retrieval service.
"""

import pytest
from app.models import (
    AddDocumentsRequest,
    AddDocumentsResponse,
    DocumentChunk,
    RetrievalRequest,
    RetrievalResponse,
)
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


class TestDocumentChunk:
    """Test cases for DocumentChunk model."""

    def test_document_chunk_valid(self):
        """Test DocumentChunk with valid data."""
        chunk = DocumentChunk(
            id="chunk_1",
            text="Test content",
            metadata={"source": "doc.pdf"},
            distance=0.5,
        )
        assert chunk.id == "chunk_1"
        assert chunk.text == "Test content"
        assert chunk.metadata == {"source": "doc.pdf"}
        assert chunk.distance == 0.5

    def test_document_chunk_negative_distance(self):
        """Test that negative distance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(id="chunk_1", text="Test content", distance=-0.1)
        assert "Input should be greater than or equal to 0" in str(exc_info.value)


class TestAddDocumentsRequest:
    """Test cases for AddDocumentsRequest model."""

    def test_add_documents_request_valid(self):
        """Test AddDocumentsRequest with valid documents."""
        request = AddDocumentsRequest(
            documents={"doc1": "content1", "doc2": "content2"}
        )
        assert request.documents == {"doc1": "content1", "doc2": "content2"}

    def test_add_documents_request_empty(self):
        """Test AddDocumentsRequest with empty documents."""
        request = AddDocumentsRequest(documents={})
        assert request.documents == {}


class TestAddDocumentsResponse:
    """Test cases for AddDocumentsResponse model."""

    def test_add_documents_response_valid(self):
        """Test AddDocumentsResponse with valid data."""
        response = AddDocumentsResponse(
            added_count=5,
            collection_name="test_collection",
            message="Successfully added documents",
        )
        assert response.added_count == 5
        assert response.collection_name == "test_collection"
        assert response.message == "Successfully added documents"

    def test_add_documents_response_negative_count(self):
        """Test that negative added_count raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AddDocumentsResponse(
                added_count=-1, collection_name="test_collection", message="test"
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)
