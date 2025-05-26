"""
Integration tests for RAG service documents endpoint.

Tests document management functionality including upload, listing, and deletion
through integration with the ingestion service.
"""

import io

from app.deps import get_http_client
from fastapi import status


class TestDocumentUpload:
    """Test document upload functionality."""

    def test_successful_pdf_upload(
        self, configured_ingestion_client, sample_pdf_content
    ):
        """Test successful PDF document upload."""
        files = {
            "file": (
                "test_document.pdf",
                io.BytesIO(sample_pdf_content),
                "application/pdf",
            )
        }

        response = configured_ingestion_client.post(
            "/api/v1/documents/upload", files=files
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "message" in data
        assert "filename" in data
        assert data["filename"] == "test_document.pdf"
        assert "status" in data

    def test_upload_with_different_filenames(
        self, configured_ingestion_client, sample_pdf_content
    ):
        """Test upload with various filename formats."""
        test_filenames = [
            "simple.pdf",
            "file with spaces.pdf",
            "file-with-hyphens.pdf",
            "file_with_underscores.pdf",
            "file.with.dots.pdf",
            "123numeric.pdf",
            "CamelCase.pdf",
        ]

        for filename in test_filenames:
            files = {
                "file": (filename, io.BytesIO(sample_pdf_content), "application/pdf")
            }

            response = configured_ingestion_client.post(
                "/api/v1/documents/upload", files=files
            )

            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert "filename" in data

    def test_upload_response_structure(
        self, configured_ingestion_client, sample_pdf_content
    ):
        """Test that upload response has correct structure."""
        files = {
            "file": (
                "test_document.pdf",
                io.BytesIO(sample_pdf_content),
                "application/pdf",
            )
        }

        response = configured_ingestion_client.post(
            "/api/v1/documents/upload", files=files
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()

        # Check required fields
        assert "message" in data
        assert "filename" in data
        assert "status" in data

        # Check field types
        assert isinstance(data["message"], str)
        assert isinstance(data["filename"], str)
        assert isinstance(data["status"], str)

    def test_multiple_file_uploads(
        self, configured_ingestion_client, sample_pdf_content
    ):
        """Test uploading multiple files sequentially."""
        filenames = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

        for filename in filenames:
            files = {
                "file": (filename, io.BytesIO(sample_pdf_content), "application/pdf")
            }

            response = configured_ingestion_client.post(
                "/api/v1/documents/upload", files=files
            )

            assert response.status_code == status.HTTP_202_ACCEPTED
            data = response.json()
            assert data["filename"] == filename


class TestDocumentUploadValidation:
    """Test document upload validation."""

    def test_upload_without_file(self, configured_ingestion_client):
        """Test upload request without file."""
        response = configured_ingestion_client.post("/api/v1/documents/upload")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_upload_empty_file(self, configured_ingestion_client):
        """Test upload with empty file."""
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}

        response = configured_ingestion_client.post(
            "/api/v1/documents/upload", files=files
        )

        # Should either accept or reject empty files based on validation
        assert response.status_code in [
            status.HTTP_202_ACCEPTED,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    def test_upload_invalid_file_type(self, configured_ingestion_client):
        """Test upload with invalid file type."""
        # Test with text file instead of PDF
        files = {
            "file": ("document.txt", io.BytesIO(b"This is a text file"), "text/plain")
        }

        response = configured_ingestion_client.post(
            "/api/v1/documents/upload", files=files
        )

        # Should either accept (if text is allowed) or reject
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        ]

    def test_upload_large_filename(
        self, configured_ingestion_client, sample_pdf_content
    ):
        """Test upload with very long filename."""
        long_filename = "a" * 300 + ".pdf"  # Very long filename

        files = {
            "file": (long_filename, io.BytesIO(sample_pdf_content), "application/pdf")
        }

        response = configured_ingestion_client.post(
            "/api/v1/documents/upload", files=files
        )

        # Should handle long filenames appropriately
        assert response.status_code in [
            status.HTTP_202_ACCEPTED,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


class TestDocumentListing:
    """Test document listing functionality."""

    def test_list_documents(self, configured_ingestion_client):
        """Test listing uploaded documents."""
        response = configured_ingestion_client.get("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "documents" in data
        assert "count" in data
        assert isinstance(data["documents"], list)
        assert isinstance(data["count"], int)

    def test_document_list_structure(self, configured_ingestion_client):
        """Test structure of document list response."""
        response = configured_ingestion_client.get("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        documents = data["documents"]
        if documents:  # If there are documents
            for doc in documents:
                assert "filename" in doc
                assert "upload_date" in doc
                assert isinstance(doc["filename"], str)
                assert isinstance(doc["upload_date"], str)

                # Optional fields
                if "size" in doc:
                    assert isinstance(doc["size"], int)
                if "pages" in doc:
                    assert isinstance(doc["pages"], int)

    def test_empty_document_list(self, integration_test_client, mocker):
        """Test listing when no documents are uploaded."""
        # Configure mock to return empty list
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"documents": [], "count": 0}
        mock_response.raise_for_status.return_value = None

        mock_http_client.get.return_value = mock_response

        response = integration_test_client.get("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["documents"] == []
        assert data["count"] == 0


class TestDocumentDeletion:
    """Test document deletion functionality."""

    def test_delete_all_documents(self, configured_ingestion_client):
        """Test deleting all documents."""
        response = configured_ingestion_client.delete("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert isinstance(data["message"], str)

    def test_delete_response_structure(self, configured_ingestion_client):
        """Test structure of delete response."""
        response = configured_ingestion_client.delete("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check required fields
        assert "message" in data

        # Optional fields that might be present
        if "deleted_documents" in data:
            assert isinstance(data["deleted_documents"], int)
        if "deleted_chunks" in data:
            assert isinstance(data["deleted_chunks"], int)

    def test_delete_when_no_documents(self, integration_test_client, mocker):
        """Test deletion when no documents exist."""
        # Configure mock for empty deletion
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "message": "No documents to delete",
            "deleted_documents": 0,
            "deleted_chunks": 0,
        }
        mock_response.raise_for_status.return_value = None

        mock_http_client.delete.return_value = mock_response

        response = integration_test_client.delete("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data


class TestDocumentStatus:
    """Test document status functionality."""

    def test_get_document_status(self, configured_ingestion_client):
        """Test getting document ingestion status."""
        response = configured_ingestion_client.get("/api/v1/documents/status")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)

    def test_status_response_structure(self, configured_ingestion_client):
        """Test structure of status response."""
        response = configured_ingestion_client.get("/api/v1/documents/status")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()

        # Check required fields for error response
        assert "detail" in data

        # For error responses, we expect a detail message
        assert isinstance(data["detail"], str)
        assert len(data["detail"]) > 0


class TestDocumentServiceIntegration:
    """Test integration with ingestion service."""

    def test_ingestion_service_upload_call(
        self, integration_test_client, sample_pdf_content, mocker
    ):
        """Test that ingestion service is called for uploads."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Configure mock response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {
            "message": "Document uploaded successfully",
            "filename": "test.pdf",
            "status": "uploaded",
        }
        mock_response.raise_for_status.return_value = None

        mock_http_client.post.return_value = mock_response

        files = {
            "file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")
        }

        response = integration_test_client.post("/api/v1/documents/upload", files=files)

        assert response.status_code == status.HTTP_202_ACCEPTED

        # Verify ingestion service was called
        assert mock_http_client.post.called

        # Check that upload URL was called
        calls = [
            call
            for call in mock_http_client.post.call_args_list
            if "upload" in str(call[0][0])
        ]
        assert len(calls) > 0

    def test_ingestion_service_list_call(self, integration_test_client, mocker):
        """Test that ingestion service is called for listing."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Configure mock response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"documents": [], "total_count": 0}
        mock_response.raise_for_status.return_value = None

        mock_http_client.get.return_value = mock_response

        response = integration_test_client.get("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK

        # Verify ingestion service was called
        assert mock_http_client.get.called

    def test_ingestion_service_delete_call(self, integration_test_client, mocker):
        """Test that ingestion service is called for deletion."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Configure mock response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"message": "Documents deleted successfully"}
        mock_response.raise_for_status.return_value = None

        mock_http_client.delete.return_value = mock_response

        response = integration_test_client.delete("/api/v1/documents")

        assert response.status_code == status.HTTP_200_OK

        # Verify ingestion service was called
        assert mock_http_client.delete.called


class TestDocumentErrorHandling:
    """Test document endpoint error handling."""

    def test_ingestion_service_upload_error(
        self, integration_test_client, sample_pdf_content, mocker
    ):
        """Test handling of ingestion service upload errors."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Configure error response
        error_response = mocker.MagicMock()
        error_response.status_code = 500
        error_response.is_success = False
        error_response.raise_for_status.side_effect = Exception(
            "Ingestion service error"
        )

        mock_http_client.post.return_value = error_response

        files = {
            "file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")
        }

        response = integration_test_client.post("/api/v1/documents/upload", files=files)

        # Should return error when ingestion service fails
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "detail" in data

    def test_ingestion_service_list_error(self, integration_test_client, mocker):
        """Test handling of ingestion service list errors."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Configure error response
        error_response = mocker.MagicMock()
        error_response.status_code = 500
        error_response.is_success = False
        error_response.raise_for_status.side_effect = Exception(
            "Ingestion service error"
        )

        mock_http_client.get.return_value = error_response

        response = integration_test_client.get("/api/v1/documents")

        # Should return error when ingestion service fails
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "detail" in data

    def test_ingestion_service_delete_error(self, integration_test_client, mocker):
        """Test handling of ingestion service delete errors."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Configure error response
        error_response = mocker.MagicMock()
        error_response.status_code = 500
        error_response.is_success = False
        error_response.raise_for_status.side_effect = Exception(
            "Ingestion service error"
        )

        mock_http_client.delete.return_value = error_response

        response = integration_test_client.delete("/api/v1/documents")

        # Should return error when ingestion service fails
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "detail" in data

    def test_network_timeout_upload(
        self, integration_test_client, sample_pdf_content, mocker
    ):
        """Test handling of network timeouts during upload."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Simulate network timeout
        mock_http_client.post.side_effect = Exception("Request timeout")

        files = {
            "file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")
        }

        response = integration_test_client.post("/api/v1/documents/upload", files=files)

        # Should handle timeout gracefully
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestDocumentWorkflow:
    """Test complete document management workflows."""

    def test_upload_list_delete_workflow(
        self, configured_ingestion_client, sample_pdf_content
    ):
        """Test complete workflow: upload -> list -> delete."""
        # Step 1: Upload document
        files = {
            "file": (
                "workflow_test.pdf",
                io.BytesIO(sample_pdf_content),
                "application/pdf",
            )
        }

        upload_response = configured_ingestion_client.post(
            "/api/v1/documents/upload", files=files
        )
        assert upload_response.status_code == status.HTTP_202_ACCEPTED

        # Step 2: List documents
        list_response = configured_ingestion_client.get("/api/v1/documents")
        assert list_response.status_code == status.HTTP_200_OK

        # Step 3: Check status
        status_response = configured_ingestion_client.get("/api/v1/documents/status")
        assert status_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

        # Step 4: Delete documents
        delete_response = configured_ingestion_client.delete("/api/v1/documents")
        assert delete_response.status_code == status.HTTP_200_OK

    def test_multiple_upload_workflow(
        self, configured_ingestion_client, sample_pdf_content
    ):
        """Test uploading multiple documents and managing them."""
        filenames = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

        # Upload multiple documents
        for filename in filenames:
            files = {
                "file": (filename, io.BytesIO(sample_pdf_content), "application/pdf")
            }

            response = configured_ingestion_client.post(
                "/api/v1/documents/upload", files=files
            )
            assert response.status_code == status.HTTP_202_ACCEPTED

        # List all documents
        list_response = configured_ingestion_client.get("/api/v1/documents")
        assert list_response.status_code == status.HTTP_200_OK

        # Check status
        status_response = configured_ingestion_client.get("/api/v1/documents/status")
        assert status_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

        # Clean up
        delete_response = configured_ingestion_client.delete("/api/v1/documents")
        assert delete_response.status_code == status.HTTP_200_OK
