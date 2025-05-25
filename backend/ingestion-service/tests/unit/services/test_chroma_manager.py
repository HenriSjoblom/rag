"""
Unit tests for ChromaDB manager services.
"""

import pytest
from app.config import Settings
from app.services.chroma_manager import (
    ChromaClientManager,
    EmbeddingModelManager,
    VectorStoreManager,
)


class TestChromaClientManager:
    """Test cases for ChromaClientManager."""

    @pytest.fixture
    def mock_settings_local(self, mocker):
        """Create mock settings for local ChromaDB."""
        settings = mocker.Mock(spec=Settings)
        settings.CHROMA_MODE = "local"
        settings.CHROMA_PATH = "/test/chroma/path"
        return settings

    @pytest.fixture
    def mock_settings_docker(self, mocker):
        """Create mock settings for Docker ChromaDB."""
        settings = mocker.Mock(spec=Settings)
        settings.CHROMA_MODE = "docker"
        settings.CHROMA_HOST = "chromadb"
        settings.CHROMA_PORT = 8000
        return settings

    def test_init_local_mode(self, mock_settings_local):
        """Test initialization in local mode."""
        manager = ChromaClientManager(mock_settings_local)
        assert manager.settings == mock_settings_local
        assert manager._client is None

    def test_init_docker_mode(self, mock_settings_docker):
        """Test initialization in Docker mode."""
        manager = ChromaClientManager(mock_settings_docker)
        assert manager.settings == mock_settings_docker
        assert manager._client is None

    def test_get_client_local_mode(self, mock_settings_local, mocker):
        """Test getting client in local mode."""
        mock_client_instance = mocker.Mock()
        mock_persistent_client = mocker.patch(
            "app.services.chroma_manager.chromadb.PersistentClient",
            return_value=mock_client_instance,
        )

        manager = ChromaClientManager(mock_settings_local)
        client = manager.get_client()

        assert client == mock_client_instance
        assert manager._client == mock_client_instance
        mock_persistent_client.assert_called_once_with(path="/test/chroma/path")

    def test_get_client_docker_mode(self, mock_settings_docker, mocker):
        """Test getting client in Docker mode."""
        mock_client_instance = mocker.Mock()
        mock_http_client = mocker.patch(
            "app.services.chroma_manager.chromadb.HttpClient",
            return_value=mock_client_instance,
        )

        manager = ChromaClientManager(mock_settings_docker)
        client = manager.get_client()

        assert client == mock_client_instance
        assert manager._client == mock_client_instance
        mock_http_client.assert_called_once_with(host="chromadb", port=8000)

    def test_get_client_cached(self, mock_settings_local, mocker):
        """Test that client is cached after first call."""
        mock_client_instance = mocker.Mock()
        mock_persistent_client = mocker.patch(
            "app.services.chroma_manager.chromadb.PersistentClient",
            return_value=mock_client_instance,
        )

        manager = ChromaClientManager(mock_settings_local)

        # First call
        client1 = manager.get_client()
        # Second call
        client2 = manager.get_client()

        assert client1 == client2
        assert client1 == mock_client_instance
        # Should only be called once due to caching
        mock_persistent_client.assert_called_once()

    def test_get_client_exception_handling(self, mock_settings_local, mocker):
        """Test exception handling when creating client."""
        mocker.patch(
            "app.services.chroma_manager.chromadb.PersistentClient",
            side_effect=Exception("Connection failed"),
        )

        manager = ChromaClientManager(mock_settings_local)

        with pytest.raises(Exception) as exc_info:
            manager.get_client()

        assert "Connection failed" in str(exc_info.value)

    def test_invalid_chroma_mode(self, mocker):
        """Test with invalid CHROMA_MODE."""
        settings = mocker.Mock(spec=Settings)
        settings.CHROMA_MODE = "invalid"

        manager = ChromaClientManager(settings)

        with pytest.raises(ValueError) as exc_info:
            manager.get_client()

        assert "Invalid CHROMA_MODE" in str(exc_info.value)


class TestEmbeddingModelManager:
    """Test cases for EmbeddingModelManager."""

    @pytest.fixture
    def mock_settings(self, mocker):
        """Create mock settings."""
        settings = mocker.Mock(spec=Settings)
        settings.EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        return settings

    def test_init(self, mock_settings):
        """Test initialization."""
        manager = EmbeddingModelManager(mock_settings)
        assert manager.settings == mock_settings
        assert manager._model is None

    def test_get_model(self, mock_settings, mocker):
        """Test getting embedding model."""
        mock_model_instance = mocker.Mock()
        mock_embedding_function = mocker.patch(
            "app.services.chroma_manager.SentenceTransformerEmbeddings",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)
        model = manager.get_model()

        assert model == mock_model_instance
        assert manager._model == mock_model_instance
        mock_embedding_function.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_get_model_cached(self, mock_settings, mocker):
        """Test that model is cached after first call."""
        mock_model_instance = mocker.Mock()
        mock_embedding_function = mocker.patch(
            "app.services.chroma_manager.SentenceTransformerEmbeddings",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)

        # First call
        model1 = manager.get_model()
        # Second call
        model2 = manager.get_model()

        assert model1 == model2
        assert model1 == mock_model_instance
        # Should only be called once due to caching
        mock_embedding_function.assert_called_once()

    def test_get_model_exception_handling(self, mock_settings, mocker):
        """Test exception handling when loading model."""
        mocker.patch(
            "app.services.chroma_manager.SentenceTransformerEmbeddings",
            side_effect=Exception("Model loading failed"),
        )

        manager = EmbeddingModelManager(mock_settings)

        with pytest.raises(Exception) as exc_info:
            manager.get_model()

        assert "Model loading failed" in str(exc_info.value)


class TestVectorStoreManager:
    """Test cases for VectorStoreManager."""

    @pytest.fixture
    def mock_settings(self, mocker):
        """Create mock settings."""
        settings = mocker.Mock(spec=Settings)
        settings.CHROMA_COLLECTION_NAME = "test_collection"
        return settings

    @pytest.fixture
    def mock_chroma_manager(self, mocker):
        """Create mock ChromaClientManager."""
        manager = mocker.Mock(spec=ChromaClientManager)
        mock_client = mocker.Mock()
        manager.get_client.return_value = mock_client
        return manager

    @pytest.fixture
    def mock_embedding_manager(self, mocker):
        """Create mock EmbeddingModelManager."""
        manager = mocker.Mock(spec=EmbeddingModelManager)
        mock_model = mocker.Mock()
        manager.get_model.return_value = mock_model
        return manager

    def test_init(self, mock_settings, mock_chroma_manager, mock_embedding_manager):
        """Test initialization."""
        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )
        assert manager.settings == mock_settings
        assert manager.chroma_manager == mock_chroma_manager
        assert manager.embedding_manager == mock_embedding_manager
        assert manager._vector_store is None

    def test_get_vector_store(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test getting vector store."""
        mock_vector_store_instance = mocker.Mock()
        mock_chroma = mocker.patch(
            "app.services.chroma_manager.Chroma",
            return_value=mock_vector_store_instance,
        )

        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_embedding_function = mocker.Mock()
        mock_embedding_manager.get_model.return_value = mock_embedding_function

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )
        vector_store = manager.get_vector_store()

        assert vector_store == mock_vector_store_instance
        assert manager._vector_store == mock_vector_store_instance
        mock_chroma.assert_called_once_with(
            client=mock_client,
            collection_name="test_collection",
            embedding_function=mock_embedding_function,
        )

    def test_get_vector_store_cached(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test that vector store is cached after first call."""
        mock_vector_store_instance = mocker.Mock()
        mock_chroma = mocker.patch(
            "app.services.chroma_manager.Chroma",
            return_value=mock_vector_store_instance,
        )

        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_embedding_function = mocker.Mock()
        mock_embedding_manager.get_model.return_value = mock_embedding_function

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # First call
        vector_store1 = manager.get_vector_store()
        # Second call
        vector_store2 = manager.get_vector_store()

        assert vector_store1 == vector_store2
        assert vector_store1 == mock_vector_store_instance
        # Should only be called once due to caching
        mock_chroma.assert_called_once()

    def test_reset(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test resetting vector store cache."""
        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # Set a cached vector store
        manager._vector_store = mocker.Mock()
        assert manager._vector_store is not None

        # Reset
        manager.reset()
        assert manager._vector_store is None

    def test_get_vector_store_after_reset(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test getting vector store after reset creates new instance."""
        mock_vector_store_instance1 = mocker.Mock()
        mock_vector_store_instance2 = mocker.Mock()
        mock_chroma = mocker.patch(
            "app.services.chroma_manager.Chroma",
            side_effect=[
                mock_vector_store_instance1,
                mock_vector_store_instance2,
            ],
        )

        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_embedding_function = mocker.Mock()
        mock_embedding_manager.get_model.return_value = mock_embedding_function

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        # First call
        vector_store1 = manager.get_vector_store()
        assert vector_store1 == mock_vector_store_instance1

        # Reset and get again
        manager.reset()
        vector_store2 = manager.get_vector_store()
        assert vector_store2 == mock_vector_store_instance2

        # Should be called twice
        assert mock_chroma.call_count == 2

    def test_get_vector_store_exception_handling(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test exception handling when creating vector store."""
        mocker.patch(
            "app.services.chroma_manager.Chroma",
            side_effect=Exception("Vector store creation failed"),
        )

        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_embedding_function = mocker.Mock()
        mock_embedding_manager.get_model.return_value = mock_embedding_function

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(Exception) as exc_info:
            manager.get_vector_store()

        assert "Vector store creation failed" in str(exc_info.value)

    def test_get_vector_store_chroma_manager_error(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager
    ):
        """Test handling error from chroma manager."""
        mock_chroma_manager.get_client.side_effect = Exception("Client error")

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(Exception) as exc_info:
            manager.get_vector_store()

        assert "Client error" in str(exc_info.value)

    def test_get_vector_store_embedding_manager_error(
        self, mock_settings, mock_chroma_manager, mock_embedding_manager, mocker
    ):
        """Test handling error from embedding manager."""
        mock_client = mocker.Mock()
        mock_chroma_manager.get_client.return_value = mock_client
        mock_embedding_manager.get_model.side_effect = Exception("Embedding error")

        manager = VectorStoreManager(
            mock_settings, mock_chroma_manager, mock_embedding_manager
        )

        with pytest.raises(Exception) as exc_info:
            manager.get_vector_store()

        assert "Embedding error" in str(exc_info.value)
