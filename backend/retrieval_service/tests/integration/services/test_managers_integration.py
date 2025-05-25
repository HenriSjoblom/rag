"""
Integration tests for manager services working together.

These tests verify that the manager services (ChromaClientManager,
EmbeddingModelManager, VectorStoreManager) work correctly when
integrated together while mocking external dependencies.
"""

import pytest
from fastapi import HTTPException


class TestManagersIntegration:
    """Integration tests for manager services working together."""

    def test_chroma_manager_integration(
        self, integration_chroma_manager, integration_settings
    ):
        """Test ChromaClientManager integration."""
        client = integration_chroma_manager.get_client()

        assert client is not None
        assert hasattr(client, "get_or_create_collection")
        assert hasattr(client, "list_collections")

    def test_embedding_manager_integration(
        self, integration_embedding_manager, integration_settings
    ):
        """Test EmbeddingModelManager integration."""
        model = integration_embedding_manager.get_model()

        assert model is not None
        assert hasattr(model, "encode")

        # Test encoding functionality
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text, convert_to_numpy=True)

        assert embedding is not None
        assert hasattr(embedding, "shape") or hasattr(embedding, "__len__")

    def test_vector_store_manager_integration(
        self, integration_vector_store_manager, integration_settings
    ):
        """Test VectorStoreManager integration."""
        collection = integration_vector_store_manager.get_collection()

        assert collection is not None
        assert hasattr(collection, "add")
        assert hasattr(collection, "query")
        assert hasattr(collection, "count")

    def test_vector_store_manager_uses_embedding_manager(
        self, integration_vector_store_manager, integration_embedding_manager
    ):
        """Test that VectorStoreManager uses EmbeddingModelManager."""
        # Get embedding model through vector store manager
        model = integration_vector_store_manager.get_embedding_model()

        assert model is not None
        # Should be the same instance as from embedding manager
        assert model == integration_embedding_manager.get_model()

    def test_managers_reset_integration(
        self,
        integration_chroma_manager,
        integration_embedding_manager,
        integration_vector_store_manager,
    ):
        """Test reset functionality across all managers."""
        # First use all managers
        integration_chroma_manager.get_client()
        integration_embedding_manager.get_model()
        integration_vector_store_manager.get_collection()

        # Reset all managers
        integration_vector_store_manager.reset()
        integration_embedding_manager.reset()
        integration_chroma_manager.reset()

        # Should be able to use them again
        client = integration_chroma_manager.get_client()
        model = integration_embedding_manager.get_model()
        collection = integration_vector_store_manager.get_collection()

        assert client is not None
        assert model is not None
        assert collection is not None

    def test_managers_error_propagation(self, integration_settings, mocker):
        """Test error propagation between managers."""
        # Test with failing ChromaDB client
        mock_failing_client = mocker.Mock()
        mock_failing_client.get_or_create_collection.side_effect = Exception(
            "Collection error"
        )

        mocker.patch("chromadb.Client", return_value=mock_failing_client)
        mocker.patch("chromadb.PersistentClient", return_value=mock_failing_client)

        from app.services.chroma_manager import ChromaClientManager
        from app.services.embedding_manager import EmbeddingModelManager
        from app.services.vector_store_manager import VectorStoreManager

        # Mock SentenceTransformer for embedding manager
        mock_model = mocker.Mock()
        mocker.patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_model
        )

        # Mock embedding function
        mock_embedding_function = mocker.Mock()
        mocker.patch(
            "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
            return_value=mock_embedding_function,
        )

        chroma_manager = ChromaClientManager(integration_settings)
        embedding_manager = EmbeddingModelManager(integration_settings)
        vector_store_manager = VectorStoreManager(
            integration_settings, chroma_manager, embedding_manager
        )

        # VectorStoreManager should propagate ChromaDB errors
        with pytest.raises(RuntimeError):
            vector_store_manager.get_collection()

    def test_managers_with_different_model_names(
        self, integration_test_data_dir, mocker
    ):
        """Test managers with different embedding model configurations."""
        from app.config import Settings
        from app.services.chroma_manager import ChromaClientManager
        from app.services.embedding_manager import EmbeddingModelManager
        from app.services.vector_store_manager import VectorStoreManager

        test_model_names = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ]

        for model_name in test_model_names:
            settings = Settings(
                EMBEDDING_MODEL_NAME=model_name,
                CHROMA_COLLECTION_NAME=f"test_collection_{model_name.replace('/', '_')}",
                CHROMA_MODE="local",
                CHROMA_PATH=str(
                    integration_test_data_dir / f"chroma_{model_name.replace('/', '_')}"
                ),
                TOP_K_RESULTS=3,
            )

            # Mock external dependencies
            mock_client = mocker.Mock()
            mock_collection = mocker.Mock()
            mock_client.get_or_create_collection.return_value = mock_collection

            mock_model = mocker.Mock()
            mock_embedding_function = mocker.Mock()

            mocker.patch("chromadb.Client", return_value=mock_client)
            mocker.patch("chromadb.PersistentClient", return_value=mock_client)
            mocker.patch(
                "sentence_transformers.SentenceTransformer", return_value=mock_model
            )
            mocker.patch(
                "app.services.vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction",
                return_value=mock_embedding_function,
            )

            # Create managers
            chroma_manager = ChromaClientManager(settings)
            embedding_manager = EmbeddingModelManager(settings)
            vector_store_manager = VectorStoreManager(
                settings, chroma_manager, embedding_manager
            )

            # Test that all managers work
            client = chroma_manager.get_client()
            model = embedding_manager.get_model()
            collection = vector_store_manager.get_collection()

            assert client is not None
            assert model is not None
            assert collection is not None


class TestManagersWithVectorSearchIntegration:
    """Integration tests for managers working with VectorSearchService."""

    def test_full_service_integration(
        self,
        integration_settings,
        integration_chroma_manager,
        integration_embedding_manager,
        integration_vector_store_manager,
    ):
        """Test full service integration with all managers."""
        from app.services.vector_search import VectorSearchService

        service = VectorSearchService(
            settings=integration_settings,
            chroma_manager=integration_chroma_manager,
            embedding_manager=integration_embedding_manager,
            vector_store_manager=integration_vector_store_manager,
        )

        assert service.settings == integration_settings
        assert service.chroma_manager == integration_chroma_manager
        assert service.embedding_manager == integration_embedding_manager
        assert service.vector_store_manager == integration_vector_store_manager

    @pytest.mark.asyncio
    async def test_service_uses_all_managers(
        self,
        integration_vector_search_service,
        integration_chroma_manager,
        integration_embedding_manager,
        integration_vector_store_manager,
    ):
        """Test that VectorSearchService uses all managers properly."""
        # Perform a search which should use all managers
        result = await integration_vector_search_service.search("test query")

        assert isinstance(result, list)

        # Verify managers are working by checking they have valid resources
        assert integration_embedding_manager.get_model() is not None
        assert integration_vector_store_manager.get_collection() is not None
        assert integration_chroma_manager.get_client() is not None

    @pytest.mark.asyncio
    async def test_service_error_handling_with_managers(
        self, integration_settings, mocker
    ):
        """Test service error handling when managers fail."""
        from app.services.chroma_manager import ChromaClientManager
        from app.services.embedding_manager import EmbeddingModelManager
        from app.services.vector_search import VectorSearchService
        from app.services.vector_store_manager import VectorStoreManager

        # Create mock managers for error testing
        mock_chroma_manager = mocker.Mock(spec=ChromaClientManager)
        mock_embedding_manager = mocker.Mock(spec=EmbeddingModelManager)
        mock_vector_store_manager = mocker.Mock(spec=VectorStoreManager)

        # Create service with mock managers
        service = VectorSearchService(
            settings=integration_settings,
            chroma_manager=mock_chroma_manager,
            embedding_manager=mock_embedding_manager,
            vector_store_manager=mock_vector_store_manager,
        )

        # Test with embedding manager failure
        mock_embedding_manager.get_model.side_effect = RuntimeError("Embedding error")

        with pytest.raises(HTTPException) as exc_info:
            await service.search("test query")
        assert exc_info.value.status_code == 500

        # Reset and test with vector store manager failure
        mock_embedding_manager.get_model.side_effect = None
        mock_embedding_manager.get_model.return_value = mocker.Mock()
        mock_vector_store_manager.get_collection.side_effect = RuntimeError(
            "Collection error"
        )

        with pytest.raises(HTTPException) as exc_info:
            await service.search("test query")
        assert exc_info.value.status_code == 503


class TestManagersResourceManagement:
    """Integration tests for resource management across managers."""

    def test_managers_share_settings_correctly(
        self,
        integration_settings,
        integration_chroma_manager,
        integration_embedding_manager,
        integration_vector_store_manager,
    ):
        """Test that all managers use the same settings instance."""
        assert integration_chroma_manager.settings == integration_settings
        assert integration_embedding_manager.settings == integration_settings
        assert integration_vector_store_manager.settings == integration_settings

    def test_managers_handle_concurrent_access(
        self,
        integration_chroma_manager,
        integration_embedding_manager,
        integration_vector_store_manager,
    ):
        """Test managers handle concurrent access patterns."""
        # Simulate concurrent access
        clients = []
        models = []
        collections = []

        for _ in range(5):
            clients.append(integration_chroma_manager.get_client())
            models.append(integration_embedding_manager.get_model())
            collections.append(integration_vector_store_manager.get_collection())

        # All should succeed
        assert len(clients) == 5
        assert len(models) == 5
        assert len(collections) == 5

        # Should be consistent (same instances if caching is implemented)
        for client in clients:
            assert client is not None
        for model in models:
            assert model is not None
        for collection in collections:
            assert collection is not None

    def test_managers_cleanup_properly(
        self,
        integration_chroma_manager,
        integration_embedding_manager,
        integration_vector_store_manager,
    ):
        """Test that managers clean up resources properly."""
        # Use all managers
        integration_chroma_manager.get_client()
        integration_embedding_manager.get_model()
        integration_vector_store_manager.get_collection()

        # Reset should not raise errors
        integration_vector_store_manager.reset()
        integration_embedding_manager.reset()
        integration_chroma_manager.reset()

        # Should be able to use again after reset
        client = integration_chroma_manager.get_client()
        model = integration_embedding_manager.get_model()
        collection = integration_vector_store_manager.get_collection()

        assert client is not None
        assert model is not None
        assert collection is not None
