"""
Integration tests for VectorSearchService.

These tests verify that VectorSearchService works correctly with real
service components while mocking external dependencies.
"""

import pytest
from app.services.vector_search import VectorSearchService
from fastapi import HTTPException


class TestVectorSearchServiceIntegration:
    """Integration tests for VectorSearchService with all dependencies."""

    @pytest.mark.asyncio
    async def test_search_with_empty_collection(
        self, integration_vector_search_service
    ):
        """Test search functionality with empty collection."""
        result = await integration_vector_search_service.search("test query")

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_search_with_populated_collection(
        self,
        integration_vector_search_service,
        mock_chroma_collection_integration,
        expected_test_documents,
    ):
        """Test search functionality with populated collection."""
        # Add documents to the collection
        collection = (
            integration_vector_search_service.vector_store_manager.get_collection()
        )
        embeddings = []
        for doc in expected_test_documents:
            embedding = (
                integration_vector_search_service.embedding_manager.get_model().encode(
                    doc
                )
            )
            embeddings.append(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            )

        collection.add(
            documents=expected_test_documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(expected_test_documents))],
            metadatas=[
                {"source": f"test_source_{i}.txt"}
                for i in range(len(expected_test_documents))
            ],
        )

        # Test search
        result = await integration_vector_search_service.search(
            "iPhone camera features"
        )

        assert isinstance(result, list)
        assert len(result) > 0

        # Verify result structure
        for item in result:
            assert "text" in item
            assert isinstance(item["text"], str)

    @pytest.mark.asyncio
    async def test_search_returns_top_k_results(
        self, integration_vector_search_service, expected_test_documents
    ):
        """Test that search returns the correct number of results."""
        # Populate collection
        collection = (
            integration_vector_search_service.vector_store_manager.get_collection()
        )
        embeddings = []
        for doc in expected_test_documents:
            embedding = (
                integration_vector_search_service.embedding_manager.get_model().encode(
                    doc
                )
            )
            embeddings.append(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            )

        collection.add(
            documents=expected_test_documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(expected_test_documents))],
            metadatas=[
                {"source": f"test_source_{i}.txt"}
                for i in range(len(expected_test_documents))
            ],
        )

        # Test with different queries
        result = await integration_vector_search_service.search("smartphone")

        # Should return up to TOP_K_RESULTS (3 in our test settings)
        assert len(result) <= integration_vector_search_service.settings.TOP_K_RESULTS

    @pytest.mark.asyncio
    async def test_search_with_various_queries(
        self,
        integration_vector_search_service,
        sample_retrieval_queries,
        expected_test_documents,
    ):
        """Test search with various query types."""
        # Populate collection
        collection = (
            integration_vector_search_service.vector_store_manager.get_collection()
        )
        embeddings = []
        for doc in expected_test_documents:
            embedding = (
                integration_vector_search_service.embedding_manager.get_model().encode(
                    doc
                )
            )
            embeddings.append(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            )

        collection.add(
            documents=expected_test_documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(expected_test_documents))],
            metadatas=[
                {"source": f"test_source_{i}.txt"}
                for i in range(len(expected_test_documents))
            ],
        )

        # Test each query
        for query in sample_retrieval_queries:
            result = await integration_vector_search_service.search(query)

            assert isinstance(result, list)
            # Each item should have proper structure
            for item in result:
                assert "text" in item
                assert isinstance(item["text"], str)
                assert len(item["text"]) > 0

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, integration_vector_search_service):
        """Test search with empty query."""
        result = await integration_vector_search_service.search("")

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_search_with_whitespace_query(
        self, integration_vector_search_service
    ):
        """Test search with whitespace-only query."""
        result = await integration_vector_search_service.search("   ")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_embedding_process(
        self, integration_vector_search_service, integration_embedding_manager
    ):
        """Test that search properly uses embedding process."""
        query = "test query for embedding"

        # This should not raise any errors and should call the embedding model
        result = await integration_vector_search_service.search(query)

        assert isinstance(result, list)

        # Verify that embedding manager is working (not asserting mock calls)
        assert integration_embedding_manager.get_model() is not None

    @pytest.mark.asyncio
    async def test_search_collection_access(
        self, integration_vector_search_service, integration_vector_store_manager
    ):
        """Test that search properly accesses the collection."""
        query = "test query for collection access"

        result = await integration_vector_search_service.search(query)

        assert isinstance(result, list)

        # Verify that vector store manager is working (not asserting mock calls)
        assert integration_vector_store_manager.get_collection() is not None

    @pytest.mark.asyncio
    async def test_search_error_handling_embedding_failure(
        self, integration_settings, mocker
    ):
        """Test error handling when embedding fails."""
        from app.services.chroma_manager import ChromaClientManager
        from app.services.embedding_manager import EmbeddingModelManager
        from app.services.vector_search import VectorSearchService
        from app.services.vector_store_manager import VectorStoreManager

        # Create mock managers for error testing
        mock_chroma_manager = mocker.Mock(spec=ChromaClientManager)
        mock_embedding_manager = mocker.Mock(spec=EmbeddingModelManager)
        mock_vector_store_manager = mocker.Mock(spec=VectorStoreManager)
        # Mock embedding manager to raise an exception
        mock_embedding_manager.get_model.side_effect = RuntimeError("Embedding failed")

        service = VectorSearchService(
            settings=integration_settings,
            chroma_manager=mock_chroma_manager,
            embedding_manager=mock_embedding_manager,
            vector_store_manager=mock_vector_store_manager,
        )

        with pytest.raises(HTTPException) as exc_info:
            await service.search("test query")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_search_error_handling_collection_failure(
        self, integration_settings, mocker
    ):
        """Test error handling when collection access fails."""
        from app.services.chroma_manager import ChromaClientManager
        from app.services.embedding_manager import EmbeddingModelManager
        from app.services.vector_store_manager import VectorStoreManager

        # Create mock managers for error testing
        mock_chroma_manager = mocker.Mock(spec=ChromaClientManager)
        mock_embedding_manager = mocker.Mock(spec=EmbeddingModelManager)
        mock_vector_store_manager = mocker.Mock(spec=VectorStoreManager)
        # Mock vector store manager to raise an exception
        mock_vector_store_manager.get_collection.side_effect = RuntimeError(
            "Collection failed"
        )

        service = VectorSearchService(
            settings=integration_settings,
            chroma_manager=mock_chroma_manager,
            embedding_manager=mock_embedding_manager,
            vector_store_manager=mock_vector_store_manager,
        )

        with pytest.raises(HTTPException) as exc_info:
            await service.search("test query")
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_get_fresh_collection_success(
        self, integration_vector_search_service
    ):
        """Test getting fresh collection instance."""
        collection = await integration_vector_search_service._get_fresh_collection()

        assert collection is not None
        assert hasattr(collection, "query")
        assert hasattr(collection, "add")

    @pytest.mark.asyncio
    async def test_get_fresh_collection_error_handling(
        self, integration_settings, mocker
    ):
        """Test error handling when getting fresh collection fails."""
        from app.services.chroma_manager import ChromaClientManager
        from app.services.embedding_manager import EmbeddingModelManager
        from app.services.vector_search import VectorSearchService
        from app.services.vector_store_manager import VectorStoreManager

        # Create mock managers for error testing
        mock_chroma_manager = mocker.Mock(spec=ChromaClientManager)
        mock_embedding_manager = mocker.Mock(spec=EmbeddingModelManager)
        mock_vector_store_manager = mocker.Mock(spec=VectorStoreManager)

        # Mock vector store manager to raise an exception
        mock_vector_store_manager.get_collection.side_effect = RuntimeError(
            "Collection access failed"
        )

        service = VectorSearchService(
            settings=integration_settings,
            chroma_manager=mock_chroma_manager,
            embedding_manager=mock_embedding_manager,
            vector_store_manager=mock_vector_store_manager,
        )

        with pytest.raises(HTTPException) as exc_info:
            await service._get_fresh_collection()
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_search_multiple_concurrent_calls(
        self, integration_vector_search_service, expected_test_documents
    ):
        """Test multiple concurrent search calls."""
        # Populate collection
        collection = (
            integration_vector_search_service.vector_store_manager.get_collection()
        )
        embeddings = []
        for doc in expected_test_documents:
            embedding = (
                integration_vector_search_service.embedding_manager.get_model().encode(
                    doc
                )
            )
            embeddings.append(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            )

        collection.add(
            documents=expected_test_documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(expected_test_documents))],
            metadatas=[
                {"source": f"test_source_{i}.txt"}
                for i in range(len(expected_test_documents))
            ],
        )

        # Run multiple searches concurrently (simulated)
        queries = ["iPhone", "battery", "camera", "iOS", "features"]
        results = []

        for query in queries:
            result = await integration_vector_search_service.search(query)
            results.append(result)

        # All searches should succeed
        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_with_special_characters(
        self, integration_vector_search_service, expected_test_documents
    ):
        """Test search with special characters and unicode."""
        # Populate collection
        collection = (
            integration_vector_search_service.vector_store_manager.get_collection()
        )
        embeddings = []
        for doc in expected_test_documents:
            embedding = (
                integration_vector_search_service.embedding_manager.get_model().encode(
                    doc
                )
            )
            embeddings.append(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            )

        collection.add(
            documents=expected_test_documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(expected_test_documents))],
            metadatas=[
                {"source": f"test_source_{i}.txt"}
                for i in range(len(expected_test_documents))
            ],
        )

        special_queries = [
            "iPhone @#$%",
            "émoji 📱 search",
            "query\nwith\nnewlines",
            "Búsqueda en español",
            "中文搜索",
        ]

        for query in special_queries:
            result = await integration_vector_search_service.search(query)
            assert isinstance(result, list)
            # Should not raise any errors


class TestVectorSearchServiceConfiguration:
    """Integration tests for VectorSearchService configuration."""

    def test_service_initialization_with_dependencies(
        self,
        integration_settings,
        integration_chroma_manager,
        integration_embedding_manager,
        integration_vector_store_manager,
    ):
        """Test service initializes correctly with all dependencies."""
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

    def test_service_uses_correct_settings(
        self, integration_vector_search_service, integration_settings
    ):
        """Test that service uses the correct settings."""
        assert (
            integration_vector_search_service.settings.TOP_K_RESULTS
            == integration_settings.TOP_K_RESULTS
        )
        assert (
            integration_vector_search_service.settings.CHROMA_COLLECTION_NAME
            == integration_settings.CHROMA_COLLECTION_NAME
        )
        assert (
            integration_vector_search_service.settings.EMBEDDING_MODEL_NAME
            == integration_settings.EMBEDDING_MODEL_NAME
        )


class TestVectorSearchServiceIntegrationEdgeCases:
    """Integration tests for edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_search_with_very_long_query(
        self, integration_vector_search_service, expected_test_documents
    ):
        """Test search with very long query."""
        # Populate collection
        collection = (
            integration_vector_search_service.vector_store_manager.get_collection()
        )
        embeddings = []
        for doc in expected_test_documents:
            embedding = (
                integration_vector_search_service.embedding_manager.get_model().encode(
                    doc
                )
            )
            embeddings.append(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            )

        collection.add(
            documents=expected_test_documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(expected_test_documents))],
            metadatas=[
                {"source": f"test_source_{i}.txt"}
                for i in range(len(expected_test_documents))
            ],
        )

        # Create a very long query
        long_query = "smartphone features " * 500  # Very long query

        result = await integration_vector_search_service.search(long_query)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_after_reset_operations(
        self, integration_vector_search_service, expected_test_documents
    ):
        """Test search functionality after reset operations."""
        # First populate and search
        collection = (
            integration_vector_search_service.vector_store_manager.get_collection()
        )
        embeddings = []
        for doc in expected_test_documents[:2]:  # Add only first 2 documents
            embedding = (
                integration_vector_search_service.embedding_manager.get_model().encode(
                    doc
                )
            )
            embeddings.append(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            )

        collection.add(
            documents=expected_test_documents[:2],
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(2)],
            metadatas=[{"source": f"test_source_{i}.txt"} for i in range(2)],
        )

        result1 = await integration_vector_search_service.search("iPhone")
        assert isinstance(result1, list)

        # Reset managers
        integration_vector_search_service.vector_store_manager.reset()
        integration_vector_search_service.embedding_manager.reset()

        # Search should still work (with empty results since we reset)
        result2 = await integration_vector_search_service.search("iPhone")
        assert isinstance(result2, list)
