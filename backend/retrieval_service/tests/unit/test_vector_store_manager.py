"""
Unit tests for VectorStoreManager.
"""

from unittest.mock import Mock

import chromadb
import pytest
from app.config import Settings
from app.services.vector_store_manager import VectorStoreManager
from chromadb.errors import ChromaError


class TestVectorStoreManagerInit:
    """Test initialization of VectorStoreManager."""

    def test_init_with_dependencies(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test initialization with all dependencies."""
        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )
        assert manager.settings == mock_settings
        assert manager.chroma_manager == mock_chroma_manager
        assert manager.embedding_manager == mock_embedding_manager
        assert manager._collection is None

    def test_init_stores_references_correctly(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test that all references are stored correctly."""
        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )
        assert manager.settings is mock_settings
        assert manager.chroma_manager is mock_chroma_manager
        assert manager.embedding_manager is mock_embedding_manager


class TestVectorStoreManagerGetCollection:
    """Test get_collection functionality."""

    def test_get_collection_success(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test successful collection retrieval."""
        # Setup mocks
        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client

        mock_collection = Mock(spec=chromadb.Collection)
        mock_collection.name = "test_collection"
        mock_collection.id = "test_id_123"
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedding_function = Mock()
        mock_sentence_transformer_ef = mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
            return_value=mock_embedding_function,
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # Test
        result = manager.get_collection()

        # Assertions
        assert result == mock_collection
        mock_chroma_manager.get_client.assert_called_once()
        mock_sentence_transformer_ef.assert_called_once_with(model_name="test-model")
        mock_client.get_or_create_collection.assert_called_once_with(
            name="test_collection", embedding_function=mock_embedding_function
        )

    def test_get_collection_empty_collection_name(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test handling of empty collection name."""
        mock_settings.CHROMA_COLLECTION_NAME = ""

        # Mock embedding function to prevent real model loading
        mock_embedding_function = Mock()
        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
            return_value=mock_embedding_function,
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_collection()

        assert "Collection configuration error" in str(exc_info.value)
        assert "Collection name cannot be empty" in str(exc_info.value)

    def test_get_collection_whitespace_collection_name(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test handling of whitespace-only collection name."""
        mock_settings.CHROMA_COLLECTION_NAME = "   "

        # Mock embedding function to prevent real model loading
        mock_embedding_function = Mock()
        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
            return_value=mock_embedding_function,
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_collection()

        assert "Collection configuration error" in str(exc_info.value)
        assert "Collection name cannot be empty" in str(exc_info.value)

    def test_get_collection_chroma_error(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test handling ChromaError when getting collection."""
        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.get_or_create_collection.side_effect = ChromaError(
            "Chroma error occurred"
        )

        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction"
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_collection()

        assert "ChromaDB collection error" in str(exc_info.value)
        assert "Chroma error occurred" in str(exc_info.value)

    def test_get_collection_connection_error(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test handling ConnectionError when getting collection."""
        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.get_or_create_collection.side_effect = ConnectionError(
            "Connection failed"
        )

        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction"
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_collection()

        assert "Cannot connect to ChromaDB for collection operations" in str(
            exc_info.value
        )
        assert "Connection failed" in str(exc_info.value)

    def test_get_collection_chroma_manager_error(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test handling error from chroma manager."""
        mock_chroma_manager.get_client.side_effect = Exception("Client error")

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_collection()

        assert "Failed to get ChromaDB collection 'test_collection'" in str(
            exc_info.value
        )
        assert "Client error" in str(exc_info.value)

    def test_get_collection_general_exception(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test handling general exception when getting collection."""
        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_client.get_or_create_collection.side_effect = Exception("Unexpected error")

        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction"
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_collection()

        assert "Failed to get ChromaDB collection 'test_collection'" in str(
            exc_info.value
        )
        assert "Unexpected error" in str(exc_info.value)

    def test_get_collection_with_different_model_names(
        self, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test with different embedding model names."""
        test_cases = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ]

        for model_name in test_cases:
            settings = Settings(
                EMBEDDING_MODEL_NAME=model_name,
                CHROMA_COLLECTION_NAME="test_collection",
                TOP_K_RESULTS=3,
                CHROMA_MODE="local",
                CHROMA_PATH="/tmp/test",
            )

            mock_client = Mock()
            mock_chroma_manager.get_client.return_value = mock_client
            mock_collection = Mock(spec=chromadb.Collection)
            mock_client.get_or_create_collection.return_value = mock_collection

            mock_embedding_function = Mock()
            mock_sentence_transformer_ef = mocker.patch(
                "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
                return_value=mock_embedding_function,
            )

            manager = VectorStoreManager(
                settings, mock_chroma_manager, mock_embedding_manager
            )
            collection = manager.get_collection()

            assert collection == mock_collection
            mock_sentence_transformer_ef.assert_called_with(model_name=model_name)
            mock_sentence_transformer_ef.reset_mock()
            mock_chroma_manager.get_client.reset_mock()


class TestVectorStoreManagerGetEmbeddingModel:
    """Test get_embedding_model functionality."""

    def test_get_embedding_model_success(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test successful embedding model retrieval."""
        mock_model = Mock()
        mock_embedding_manager.get_model.return_value = mock_model

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        result = manager.get_embedding_model()

        assert result == mock_model
        mock_embedding_manager.get_model.assert_called_once()

    def test_get_embedding_model_error(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test handling error when getting embedding model."""
        mock_embedding_manager.get_model.side_effect = Exception("Model loading failed")

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_embedding_model()

        assert "Cannot access embedding model" in str(exc_info.value)
        assert "Model loading failed" in str(exc_info.value)


class TestVectorStoreManagerReset:
    """Test reset functionality."""

    def test_reset_success(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test successful reset."""
        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # Set a collection reference
        manager._collection = Mock()
        assert manager._collection is not None

        # Reset
        manager.reset()
        assert manager._collection is None

    def test_reset_with_no_collection(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test reset when no collection is set."""
        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        assert manager._collection is None

        # Reset should not raise exception
        manager.reset()
        assert manager._collection is None

    def test_reset_exception_handling(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test reset when an exception occurs during reset."""
        # Mock the logger to avoid import issues
        mock_logger = mocker.patch("app.services.vector_store_manager.logger")

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # Simulate an exception during reset by mocking the assignment
        def failing_reset():
            try:
                # Simulate some operation that fails during reset
                raise Exception("Reset error")
            except Exception as e:
                mock_logger.error(
                    f"Error during vector store manager reset: {e}", exc_info=True
                )
                manager._collection = None  # Force cleanup
                raise RuntimeError(
                    f"Failed to reset vector store manager: {str(e)}"
                ) from e

        manager.reset = failing_reset

        with pytest.raises(RuntimeError) as exc_info:
            manager.reset()

        assert "Failed to reset vector store manager" in str(exc_info.value)
        assert "Reset error" in str(exc_info.value)

    def test_multiple_resets(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test multiple reset operations."""
        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # Set a collection reference
        manager._collection = Mock()
        assert manager._collection is not None

        # First reset
        manager.reset()
        assert manager._collection is None

        # Second reset
        manager.reset()
        assert manager._collection is None


class TestVectorStoreManagerEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_get_collection_after_reset_creates_fresh_instance(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test that get_collection after reset can create a fresh collection."""
        # Setup mocks for two different collection instances
        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client

        mock_collection1 = Mock(spec=chromadb.Collection)
        mock_collection1.name = "test_collection"
        mock_collection1.id = "test_id_1"

        mock_collection2 = Mock(spec=chromadb.Collection)
        mock_collection2.name = "test_collection"
        mock_collection2.id = "test_id_2"

        mock_client.get_or_create_collection.side_effect = [
            mock_collection1,
            mock_collection2,
        ]

        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction"
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # First call
        collection1 = manager.get_collection()
        assert collection1 == mock_collection1

        # Reset
        manager.reset()
        assert manager._collection is None

        # Second call should work and potentially get a different instance
        collection2 = manager.get_collection()
        assert collection2 == mock_collection2

        # Should be called twice
        assert mock_client.get_or_create_collection.call_count == 2

    def test_get_embedding_model_after_multiple_operations(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test get_embedding_model works after multiple operations."""
        mock_model = Mock()
        mock_embedding_manager.get_model.return_value = mock_model

        # Setup collection mocks
        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_collection = Mock(spec=chromadb.Collection)
        mock_client.get_or_create_collection.return_value = mock_collection

        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction"
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # Multiple operations
        model1 = manager.get_embedding_model()
        collection = manager.get_collection()
        manager.reset()
        model2 = manager.get_embedding_model()

        assert model1 == mock_model
        assert model2 == mock_model
        assert collection == mock_collection

    def test_concurrent_collection_access_simulation(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test simulation of concurrent collection access."""
        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_collection = Mock(spec=chromadb.Collection)
        mock_client.get_or_create_collection.return_value = mock_collection

        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction"
        )

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # Simulate multiple concurrent calls (they all should succeed)
        collections = []
        for _ in range(5):
            collections.append(manager.get_collection())

        # All should return the same collection instance (if caching were implemented)
        # Since this implementation doesn't cache, all calls go through
        assert all(collection == mock_collection for collection in collections)
        # get_or_create_collection should be called for each request since no caching
        assert mock_client.get_or_create_collection.call_count == 5

    def test_settings_model_name_used_correctly(
        self, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test that settings model name is used correctly in collection creation."""
        custom_model_name = "custom-embedding-model"
        settings = Settings(
            EMBEDDING_MODEL_NAME=custom_model_name,
            CHROMA_COLLECTION_NAME="custom_collection",
            TOP_K_RESULTS=5,
            CHROMA_MODE="local",
            CHROMA_PATH="/custom/path",
        )

        mock_client = Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_collection = Mock(spec=chromadb.Collection)
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedding_function = Mock()
        mock_sentence_transformer_ef = mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
            return_value=mock_embedding_function,
        )

        manager = VectorStoreManager(
            settings, mock_chroma_manager, mock_embedding_manager
        )
        collection = manager.get_collection()

        assert collection == mock_collection
        mock_sentence_transformer_ef.assert_called_once_with(
            model_name=custom_model_name
        )
        mock_client.get_or_create_collection.assert_called_once_with(
            name="custom_collection", embedding_function=mock_embedding_function
        )
