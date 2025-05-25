"""
Unit tests for the CollectionManagerService.
"""

import pytest
from app.config import Settings
from app.services.chroma_manager import ChromaClientManager, VectorStoreManager
from app.services.collection_manager import CollectionManagerService
from app.services.file_management import FileManagementService


class TestCollectionManagerService:
    """Test cases for CollectionManagerService."""

    @pytest.fixture
    def mock_settings(self, mocker):
        """Create mock settings."""
        settings = mocker.Mock(spec=Settings)
        settings.CHROMA_COLLECTION_NAME = "test_collection"
        settings.SOURCE_DIRECTORY = "/test/source"
        return settings

    @pytest.fixture
    def mock_chroma_manager(self, mocker):
        """Create mock ChromaClientManager."""
        manager = mocker.Mock(spec=ChromaClientManager)
        mock_client = mocker.Mock()
        manager.get_client.return_value = mock_client
        return manager

    @pytest.fixture
    def mock_vector_store_manager(self, mocker):
        """Create mock VectorStoreManager."""
        return mocker.Mock(spec=VectorStoreManager)

    @pytest.fixture
    def collection_service(
        self,
        mock_settings,
        mock_chroma_manager,
        mock_vector_store_manager,
        mocker,
    ):
        """Create CollectionManagerService instance."""
        # Mock the FileManagementService since it's created internally
        mock_file_service = mocker.Mock(spec=FileManagementService)
        mocker.patch(
            "app.services.collection_manager.FileManagementService",
            return_value=mock_file_service,
        )

        service = CollectionManagerService(
            mock_settings,
            mock_chroma_manager,
            mock_vector_store_manager,
        )
        # Store the mock file service for test access
        service._mock_file_service = mock_file_service
        return service

    def test_init(
        self,
        mock_settings,
        mock_chroma_manager,
        mock_vector_store_manager,
        mocker,
    ):
        """Test CollectionManagerService initialization."""
        # Mock the FileManagementService since it's created internally
        mock_file_service = mocker.Mock(spec=FileManagementService)
        mocker.patch(
            "app.services.collection_manager.FileManagementService",
            return_value=mock_file_service,
        )

        service = CollectionManagerService(
            mock_settings,
            mock_chroma_manager,
            mock_vector_store_manager,
        )

        assert service.settings == mock_settings
        assert service.chroma_manager == mock_chroma_manager
        assert service.vector_store_manager == mock_vector_store_manager
        assert service.file_service == mock_file_service

    def test_clear_collection_and_documents_success(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test successful clearing of collection and documents."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None
        collection_service._mock_file_service.clear_all_files.return_value = 5

        result = collection_service.clear_all()

        assert result["collection_deleted"] is True
        assert result["source_files_cleared"] is True
        assert result["files_deleted_count"] == 5
        assert result["overall_success"] is True
        assert any("deleted successfully" in msg for msg in result["messages"])
        assert any("Cleared 5 files" in msg for msg in result["messages"])

        # Verify calls
        mock_chroma_manager.get_client.assert_called_once()
        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once_with("test_collection")
        collection_service._mock_file_service.clear_all_files.assert_called_once()

    def test_clear_collection_not_found(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test clearing collection when collection doesn't exist."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.side_effect = Exception(
            "Collection 'test_collection' does not exist"
        )
        mock_client.create_collection.return_value = None
        collection_service._mock_file_service.clear_all_files.return_value = 3

        result = collection_service.clear_all()

        assert result["collection_deleted"] is True
        assert result["source_files_cleared"] is True
        assert result["files_deleted_count"] == 3
        assert result["overall_success"] is True
        assert any("not found" in msg for msg in result["messages"])
        assert any("Cleared 3 files" in msg for msg in result["messages"])

    def test_clear_collection_chroma_error(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test clearing collection when ChromaDB error occurs."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.side_effect = Exception("Connection error")
        collection_service._mock_file_service.clear_all_files.return_value = 2

        result = collection_service.clear_all()

        assert result["collection_deleted"] is False
        assert result["source_files_cleared"] is True
        assert result["files_deleted_count"] == 2
        assert result["overall_success"] is False
        assert any(
            "Failed to manage ChromaDB collection" in msg for msg in result["messages"]
        )
        assert any("Connection error" in msg for msg in result["messages"])

    def test_clear_collection_file_service_error(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test clearing collection when file service error occurs."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None
        collection_service._mock_file_service.clear_all_files.side_effect = (
            RuntimeError("File system error")
        )

        result = collection_service.clear_all()

        assert result["collection_deleted"] is True
        assert result["source_files_cleared"] is False
        assert result["files_deleted_count"] == 0
        assert result["overall_success"] is False
        assert any("deleted successfully" in msg for msg in result["messages"])
        assert any("Failed to clear source files" in msg for msg in result["messages"])
        assert any("File system error" in msg for msg in result["messages"])

    def test_clear_collection_both_operations_fail(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test clearing when both collection deletion and file clearing fail."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.side_effect = Exception("DB error")
        collection_service._mock_file_service.clear_all_files.side_effect = (
            RuntimeError("FS error")
        )

        result = collection_service.clear_all()

        assert result["collection_deleted"] is False
        assert result["source_files_cleared"] is False
        assert result["files_deleted_count"] == 0
        assert result["overall_success"] is False
        assert any(
            "Failed to manage ChromaDB collection" in msg for msg in result["messages"]
        )
        assert any("Failed to clear source files" in msg for msg in result["messages"])

    def test_clear_collection_chroma_manager_error(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test clearing when getting ChromaDB client fails."""
        # Setup mocks
        mock_chroma_manager.get_client.side_effect = Exception(
            "Client connection failed"
        )
        collection_service._mock_file_service.clear_all_files.return_value = 1

        result = collection_service.clear_all()

        assert result["collection_deleted"] is False
        assert result["source_files_cleared"] is True
        assert result["files_deleted_count"] == 1
        assert result["overall_success"] is False
        assert any(
            "Failed to manage ChromaDB collection" in msg for msg in result["messages"]
        )
        assert any("Client connection failed" in msg for msg in result["messages"])

    def test_clear_collection_vector_store_reset(
        self,
        collection_service,
        mock_chroma_manager,
        mock_vector_store_manager,
        mocker,
    ):
        """Test that vector store manager is reset after successful collection deletion."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None
        collection_service._mock_file_service.clear_all_files.return_value = 0

        result = collection_service.clear_all()

        assert result["collection_deleted"] is True
        mock_vector_store_manager.reset.assert_called_once()

    def test_clear_collection_vector_store_not_reset_on_failure(
        self,
        collection_service,
        mock_chroma_manager,
        mock_vector_store_manager,
        mocker,
    ):
        """Test that vector store manager is not reset when collection deletion fails."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.side_effect = Exception("Delete failed")
        collection_service._mock_file_service.clear_all_files.return_value = 0

        result = collection_service.clear_all()

        assert result["collection_deleted"] is False
        mock_vector_store_manager.reset.assert_not_called()

    def test_clear_collection_empty_source_directory(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test clearing collection with empty source directory."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None
        collection_service._mock_file_service.clear_all_files.return_value = 0

        result = collection_service.clear_all()

        assert result["collection_deleted"] is True
        assert result["source_files_cleared"] is True
        assert result["files_deleted_count"] == 0
        assert result["overall_success"] is True
        assert any("deleted successfully" in msg for msg in result["messages"])
        assert any("Cleared 0 files" in msg for msg in result["messages"])

    def test_clear_collection_messages_format(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test that the messages in the result are properly formatted."""
        # Setup mocks for successful operation
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None
        collection_service._mock_file_service.clear_all_files.return_value = 10

        result = collection_service.clear_all()

        messages = result["messages"]
        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert all(isinstance(msg, str) for msg in messages)
        assert any("collection" in msg.lower() for msg in messages)
        assert any("files" in msg.lower() for msg in messages)

    def test_clear_collection_return_structure(
        self, collection_service, mock_chroma_manager, mocker
    ):
        """Test that the return structure contains all expected keys."""
        # Setup mocks
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None
        collection_service._mock_file_service.clear_all_files.return_value = 7

        result = collection_service.clear_all()

        expected_keys = {
            "collection_deleted",
            "source_files_cleared",
            "files_deleted_count",
            "messages",
            "overall_success",
        }
        assert set(result.keys()) == expected_keys
        assert isinstance(result["collection_deleted"], bool)
        assert isinstance(result["source_files_cleared"], bool)
        assert isinstance(result["files_deleted_count"], int)
        assert isinstance(result["messages"], list)
        assert isinstance(result["overall_success"], bool)
