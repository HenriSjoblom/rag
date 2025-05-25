"""
Unit tests for Pydantic models in the ingestion service.
"""

import pytest
from app.models import (
    DocumentDetail,
    DocumentListResponse,
    IngestionResponse,
    IngestionStatus,
    IngestionStatusResponse,
)
from pydantic import ValidationError


class TestIngestionStatus:
    """Test cases for IngestionStatus model."""

    def test_ingestion_status_default_values(self):
        """Test that IngestionStatus has correct default values."""
        status = IngestionStatus()
        assert status.documents_processed == 0
        assert status.chunks_added == 0
        assert status.errors == []

    def test_ingestion_status_valid_values(self):
        """Test IngestionStatus with valid values."""
        status = IngestionStatus(
            documents_processed=5, chunks_added=100, errors=["Error 1", "Error 2"]
        )
        assert status.documents_processed == 5
        assert status.chunks_added == 100
        assert status.errors == ["Error 1", "Error 2"]

    def test_ingestion_status_negative_documents_processed(self):
        """Test that negative documents_processed raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionStatus(documents_processed=-1)
        assert "Input should be greater than or equal to 0" in str(exc_info.value)

    def test_ingestion_status_negative_chunks_added(self):
        """Test that negative chunks_added raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionStatus(chunks_added=-1)
        assert "Input should be greater than or equal to 0" in str(exc_info.value)

    def test_ingestion_status_serialization(self):
        """Test that IngestionStatus can be serialized to dict."""
        status = IngestionStatus(
            documents_processed=3, chunks_added=50, errors=["Test error"]
        )
        data = status.model_dump()
        expected = {
            "documents_processed": 3,
            "chunks_added": 50,
            "errors": ["Test error"],
        }
        assert data == expected


class TestIngestionResponse:
    """Test cases for IngestionResponse model."""

    def test_ingestion_response_required_fields(self):
        """Test that IngestionResponse requires status field."""
        response = IngestionResponse(status="completed")
        assert response.status == "completed"
        assert response.documents_found is None
        assert response.message is None

    def test_ingestion_response_all_fields(self):
        """Test IngestionResponse with all fields."""
        response = IngestionResponse(
            status="processing", documents_found=10, message="Processing in progress"
        )
        assert response.status == "processing"
        assert response.documents_found == 10
        assert response.message == "Processing in progress"

    def test_ingestion_response_empty_status(self):
        """Test that empty status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionResponse(status="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_ingestion_response_negative_documents_found(self):
        """Test that negative documents_found raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionResponse(status="completed", documents_found=-1)
        assert "Input should be greater than or equal to 0" in str(exc_info.value)


class TestDocumentDetail:
    """Test cases for DocumentDetail model."""

    def test_document_detail_valid_name(self):
        """Test DocumentDetail with valid name."""
        doc = DocumentDetail(name="test_document.pdf")
        assert doc.name == "test_document.pdf"

    def test_document_detail_empty_name(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentDetail(name="")
        assert "String should have at least 1 character" in str(exc_info.value)


class TestDocumentListResponse:
    """Test cases for DocumentListResponse model."""

    def test_document_list_response_valid(self):
        """Test DocumentListResponse with valid data."""
        docs = [DocumentDetail(name="doc1.pdf"), DocumentDetail(name="doc2.pdf")]
        response = DocumentListResponse(count=2, documents=docs)
        assert response.count == 2
        assert len(response.documents) == 2
        assert response.documents[0].name == "doc1.pdf"
        assert response.documents[1].name == "doc2.pdf"

    def test_document_list_response_empty_list(self):
        """Test DocumentListResponse with empty document list."""
        response = DocumentListResponse(count=0, documents=[])
        assert response.count == 0
        assert response.documents == []

    def test_document_list_response_count_mismatch(self):
        """Test that count mismatch raises ValidationError."""
        docs = [DocumentDetail(name="doc1.pdf")]
        with pytest.raises(ValidationError) as exc_info:
            DocumentListResponse(count=2, documents=docs)
        assert "Count must match number of documents" in str(exc_info.value)

    def test_document_list_response_negative_count(self):
        """Test that negative count raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentListResponse(count=-1, documents=[])
        assert "Input should be greater than or equal to 0" in str(exc_info.value)


class TestIngestionStatusResponse:
    """Test cases for IngestionStatusResponse model."""

    def test_ingestion_status_response_minimal(self):
        """Test IngestionStatusResponse with minimal required fields."""
        response = IngestionStatusResponse(is_processing=True, status="running")
        assert response.is_processing is True
        assert response.status == "running"
        assert response.last_completed is None
        assert response.documents_processed is None
        assert response.chunks_added is None
        assert response.errors is None

    def test_ingestion_status_response_full(self):
        """Test IngestionStatusResponse with all fields."""
        response = IngestionStatusResponse(
            is_processing=False,
            status="completed",
            last_completed="2025-01-01T12:00:00Z",
            documents_processed=5,
            chunks_added=100,
            errors=["Warning: Large file processed"],
        )
        assert response.is_processing is False
        assert response.status == "completed"
        assert response.last_completed == "2025-01-01T12:00:00Z"
        assert response.documents_processed == 5
        assert response.chunks_added == 100
        assert response.errors == ["Warning: Large file processed"]

    def test_ingestion_status_response_empty_status(self):
        """Test that empty status raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionStatusResponse(is_processing=False, status="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_ingestion_status_response_negative_values(self):
        """Test that negative values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IngestionStatusResponse(
                is_processing=False, status="completed", documents_processed=-1
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            IngestionStatusResponse(
                is_processing=False, status="completed", chunks_added=-1
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)

    def test_ingestion_status_response_serialization(self):
        """Test that IngestionStatusResponse can be serialized to dict."""
        response = IngestionStatusResponse(
            is_processing=True,
            status="processing",
            documents_processed=3,
            chunks_added=75,
        )
        data = response.model_dump()
        expected = {
            "is_processing": True,
            "status": "processing",
            "last_completed": None,
            "documents_processed": 3,
            "chunks_added": 75,
            "errors": None,
        }
        assert data == expected
