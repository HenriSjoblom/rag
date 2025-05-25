"""
Unit tests for VectorSearchService in the retrieval service.
"""

from unittest.mock import MagicMock

import pytest
from app.services.vector_search import VectorSearchService
from fastapi import HTTPException

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
def vector_search_service(mock_managers):
    """Create VectorSearchService with mocked dependencies."""
    return VectorSearchService(
        settings=mock_managers["settings"],
        chroma_manager=mock_managers["chroma_manager"],
        embedding_manager=mock_managers["embedding_manager"],
        vector_store_manager=mock_managers["vector_store_manager"],
    )


class TestVectorSearchService:
    """Test cases for VectorSearchService."""

    async def test_embed_query_success(self, vector_search_service, mock_managers):
        """Test successful query embedding."""
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.ndim = 1
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

        mock_managers["embedding_manager"].get_model.return_value = mock_model

        query = "test query"
        result = vector_search_service._embed_query(query)

        mock_managers["embedding_manager"].get_model.assert_called_once()
        mock_model.encode.assert_called_once_with(query.strip(), convert_to_numpy=True)
        assert result == [0.1, 0.2, 0.3]

    async def test_embed_query_empty_query(self, vector_search_service):
        """Test embedding with empty query raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            vector_search_service._embed_query("")

        assert exc_info.value.status_code == 400
        assert "Query cannot be empty" in exc_info.value.detail

    async def test_embed_query_whitespace_only(self, vector_search_service):
        """Test embedding with whitespace-only query raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            vector_search_service._embed_query("   ")

        assert exc_info.value.status_code == 400
        assert "Query cannot be empty" in exc_info.value.detail

    async def test_embed_query_model_error(self, vector_search_service, mock_managers):
        """Test embedding when model raises an error."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_managers["embedding_manager"].get_model.return_value = mock_model

        with pytest.raises(HTTPException) as exc_info:
            vector_search_service._embed_query("test query")

        assert exc_info.value.status_code == 500
        assert "Failed to generate query embedding" in exc_info.value.detail

    async def test_search_success(self, vector_search_service, mock_managers, mocker):
        """Test successful search operation."""
        # Mock embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.ndim = 1
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_managers["embedding_manager"].get_model.return_value = mock_model

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.id = "test_id"
        mock_collection.query.return_value = {
            "ids": [["doc_1", "doc_2"]],
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[{"source": "test1.pdf"}, {"source": "test2.pdf"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_managers[
            "vector_store_manager"
        ].get_collection.return_value = mock_collection

        # Mock asyncio.to_thread for collection access
        mock_to_thread = mocker.patch("app.services.vector_search.asyncio.to_thread")
        mock_to_thread.side_effect = [
            mock_collection,
            mock_collection.query.return_value,
        ]

        result = await vector_search_service.search("test query")

        assert len(result) == 2
        assert result[0]["id"] == "doc_1"
        assert result[0]["text"] == "Document 1 content"
        assert result[0]["metadata"] == {"source": "test1.pdf"}
        assert result[0]["distance"] == 0.1

    async def test_search_empty_query_returns_empty(self, vector_search_service):
        """Test that empty query returns empty results."""
        result = await vector_search_service.search("")
        assert result == []

    async def test_search_no_results(
        self, vector_search_service, mock_managers, mocker
    ):
        """Test search when no documents are found."""
        # Mock embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.ndim = 1
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_managers["embedding_manager"].get_model.return_value = mock_model

        # Mock collection with empty results
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.id = "test_id"
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_managers[
            "vector_store_manager"
        ].get_collection.return_value = mock_collection

        mock_to_thread = mocker.patch("app.services.vector_search.asyncio.to_thread")
        mock_to_thread.side_effect = [
            mock_collection,
            mock_collection.query.return_value,
        ]

        result = await vector_search_service.search("no matches")

        assert result == []

    async def test_search_database_error(
        self, vector_search_service, mock_managers, mocker
    ):
        """Test search when database raises an error."""
        # Mock embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.ndim = 1
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_managers["embedding_manager"].get_model.return_value = mock_model

        # Mock collection that raises error
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.id = "test_id"
        mock_collection.query.side_effect = Exception("Database connection failed")
        mock_managers[
            "vector_store_manager"
        ].get_collection.return_value = mock_collection

        mock_to_thread = mocker.patch("app.services.vector_search.asyncio.to_thread")
        mock_to_thread.side_effect = [
            mock_collection,
            Exception("Database connection failed"),
        ]

        with pytest.raises(HTTPException) as exc_info:
            await vector_search_service.search("test query")

        assert exc_info.value.status_code == 500
        assert "Failed to query vector database" in exc_info.value.detail

    async def test_get_fresh_collection_success(
        self, vector_search_service, mock_managers, mocker
    ):
        """Test successful collection retrieval."""
        mock_collection = MagicMock()
        mock_managers[
            "vector_store_manager"
        ].get_collection.return_value = mock_collection

        mock_to_thread = mocker.patch("app.services.vector_search.asyncio.to_thread")
        mock_to_thread.return_value = mock_collection

        result = await vector_search_service._get_fresh_collection()

        assert result == mock_collection
        mock_to_thread.assert_called_once()

    async def test_get_fresh_collection_error(
        self, vector_search_service, mock_managers, mocker
    ):
        """Test collection retrieval when it fails."""
        mock_managers["vector_store_manager"].get_collection.side_effect = RuntimeError(
            "Collection not available"
        )

        mock_to_thread = mocker.patch("app.services.vector_search.asyncio.to_thread")
        mock_to_thread.side_effect = RuntimeError("Collection not available")

        with pytest.raises(HTTPException) as exc_info:
            await vector_search_service._get_fresh_collection()

        assert exc_info.value.status_code == 503
        assert "Vector database collection not available" in exc_info.value.detail


class TestVectorSearchServiceEdgeCases:
    """Test edge cases and error conditions."""

    async def test_search_malformed_results(
        self, vector_search_service, mock_managers, mocker
    ):
        """Test search with malformed results from ChromaDB."""
        # Mock embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.ndim = 1
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_managers["embedding_manager"].get_model.return_value = mock_model

        # Mock collection with malformed results
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.id = "test_id"
        mock_collection.query.return_value = {
            "ids": [["doc_1", "doc_2", "doc_3"]],  # 3 IDs
            "documents": [["Document 1", "Document 2"]],  # Only 2 documents
            "metadatas": [[{"source": "test1.pdf"}]],  # Only 1 metadata
            "distances": [[0.1, 0.2, 0.3]],  # 3 distances
        }
        mock_managers[
            "vector_store_manager"
        ].get_collection.return_value = mock_collection

        mock_to_thread = mocker.patch("app.services.vector_search.asyncio.to_thread")
        mock_to_thread.side_effect = [
            mock_collection,
            mock_collection.query.return_value,
        ]

        result = await vector_search_service.search("test query")

        # Should handle malformed data gracefully, only return valid entries
        valid_results = [r for r in result if r["id"] and r["text"]]
        assert len(valid_results) <= 2  # At most 2 valid results due to document count

    async def test_search_zero_top_k(self, mock_managers, mocker):
        """Test search with invalid TOP_K_RESULTS configuration."""
        mock_managers["settings"].TOP_K_RESULTS = 0

        service = VectorSearchService(
            settings=mock_managers["settings"],
            chroma_manager=mock_managers["chroma_manager"],
            embedding_manager=mock_managers["embedding_manager"],
            vector_store_manager=mock_managers["vector_store_manager"],
        )

        # Mock embedding
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.ndim = 1
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_managers["embedding_manager"].get_model.return_value = mock_model

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.id = "test_id"
        mock_managers[
            "vector_store_manager"
        ].get_collection.return_value = mock_collection

        mock_to_thread = mocker.patch("app.services.vector_search.asyncio.to_thread")
        mock_to_thread.side_effect = [
            mock_collection,
            ValueError("TOP_K_RESULTS must be greater than 0"),
        ]

        with pytest.raises(HTTPException) as exc_info:
            await service.search("test query")

        assert exc_info.value.status_code == 500
        assert "Search configuration error" in exc_info.value.detail
