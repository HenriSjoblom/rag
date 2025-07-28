"""
Service integration tests for ingestion service.
Tests service components with minimal dependencies.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


class TestStateServiceIntegration:
    """Test ingestion state service integration."""

    def test_state_service_initialization(self, mock_state_service):
        """Test that state service initializes properly."""
        status = mock_state_service.get_status()
        assert status["is_processing"] is False
        assert status["status"] == "idle"
        assert status["documents_processed"] == 0

    def test_state_service_start_ingestion(self, mock_state_service):
        """Test starting ingestion process."""
        mock_state_service.is_ingesting.return_value = False
        result = mock_state_service.start_ingestion()
        assert result is True
        mock_state_service.start_ingestion.assert_called_once()

    def test_state_service_already_ingesting(self, mock_state_service):
        """Test behavior when ingestion is already running."""
        mock_state_service.is_ingesting.return_value = True
        # Should not start new ingestion
        assert mock_state_service.is_ingesting() is True


class TestFileServiceIntegration:
    """Test file management service integration."""

    def test_file_service_list_documents(self, mock_file_service):
        """Test listing documents from file service."""
        docs = mock_file_service.list_documents()
        assert len(docs) == 2
        assert docs[0]["name"] == "document1.pdf"
        assert docs[1]["name"] == "document2.pdf"

    def test_file_service_count_documents(self, mock_file_service):
        """Test counting documents."""
        count = mock_file_service.count_documents()
        assert count == 2

    def test_file_service_upload_handling(self, mock_file_service):
        """Test file upload functionality."""
        # Test non-duplicate upload
        mock_file_service.has_duplicate_filename.return_value = False
        mock_file_service.save_uploaded_file.return_value = ("/tmp/test.pdf", False)

        path, is_duplicate = mock_file_service.save_uploaded_file("test.pdf", b"content")
        assert path == "/tmp/test.pdf"
        assert is_duplicate is False

        # Test duplicate detection
        mock_file_service.has_duplicate_filename.return_value = True
        assert mock_file_service.has_duplicate_filename("duplicate.pdf") is True


class TestCollectionServiceIntegration:
    """Test collection management service integration."""

    def test_collection_service_clear(self, mock_collection_service):
        """Test collection clearing functionality."""
        result = mock_collection_service.clear_collection_and_documents()

        assert result["collection_cleared"] is True
        assert result["documents_cleared"] is True
        assert len(result["messages"]) == 2
        assert "Collection cleared successfully" in result["messages"]

    def test_collection_service_partial_failure(self, mock_collection_service):
        """Test partial failure scenarios."""
        # Configure partial failure
        mock_collection_service.clear_collection_and_documents.return_value = {
            "collection_cleared": True,
            "documents_cleared": False,
            "messages": [
                "Collection cleared successfully",
                "Failed to clear documents: Database error"
            ]
        }

        result = mock_collection_service.clear_collection_and_documents()
        assert result["collection_cleared"] is True
        assert result["documents_cleared"] is False


class TestIngestionProcessorIntegration:
    """Test ingestion processor integration."""

    def test_processor_initialization(self, mock_ingestion_processor):
        """Test processor initializes correctly."""
        assert mock_ingestion_processor is not None

        # Test default status
        processed_files = mock_ingestion_processor._get_processed_files()
        assert isinstance(processed_files, set)
        assert len(processed_files) == 0

    def test_processor_run_ingestion(self, mock_ingestion_processor):
        """Test running ingestion process."""
        status = mock_ingestion_processor.run_ingestion()

        assert status.documents_processed == 2
        assert status.chunks_added == 10
        assert len(status.errors) == 0

    def test_processor_with_errors(self, mock_ingestion_processor, mocker):
        """Test processor handling errors."""
        # Configure error scenario
        error_status = mocker.Mock()
        error_status.documents_processed = 1
        error_status.chunks_added = 5
        error_status.errors = ["Failed to process document2.pdf: Invalid format"]

        mock_ingestion_processor.run_ingestion.return_value = error_status

        status = mock_ingestion_processor.run_ingestion()
        assert status.documents_processed == 1
        assert len(status.errors) == 1
        assert "Invalid format" in status.errors[0]


class TestServiceCombination:
    """Test services working together."""

    def test_complete_ingestion_workflow(self, mock_state_service, mock_file_service,
                                       mock_collection_service, mock_ingestion_processor):
        """Test complete ingestion workflow with all services."""
        # 1. Check initial state
        assert mock_state_service.is_ingesting() is False

        # 2. Check documents are available
        count = mock_file_service.count_documents()
        assert count > 0

        # 3. Start ingestion
        mock_state_service.is_ingesting.return_value = False
        result = mock_state_service.start_ingestion()
        assert result is True

        # 4. Run processing
        status = mock_ingestion_processor.run_ingestion()
        assert status.documents_processed > 0

        # 5. Verify final state
        final_status = mock_state_service.get_status()
        assert "documents_processed" in final_status

    def test_upload_and_process_workflow(self, mock_file_service, mock_state_service):
        """Test uploading a file and then processing it."""
        # 1. Upload file
        mock_file_service.has_duplicate_filename.return_value = False
        path, is_duplicate = mock_file_service.save_uploaded_file("new.pdf", b"content")
        assert not is_duplicate

        # 2. Update document count
        mock_file_service.count_documents.return_value = 3  # One more document

        # 3. Start ingestion
        assert mock_state_service.is_ingesting() is False
        result = mock_state_service.start_ingestion()
        assert result is True

    def test_clear_and_rebuild_workflow(self, mock_collection_service, mock_state_service):
        """Test clearing collection and rebuilding."""
        # 1. Ensure not processing
        assert mock_state_service.is_ingesting() is False

        # 2. Clear collection
        result = mock_collection_service.clear_collection_and_documents()
        assert result["collection_cleared"] is True

        # 3. Start fresh ingestion
        new_result = mock_state_service.start_ingestion()
        assert new_result is True