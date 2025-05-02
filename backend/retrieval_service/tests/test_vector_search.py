import pytest
from unittest.mock import MagicMock, AsyncMock, ANY # Import ANY
from fastapi import HTTPException

from app.services.vector_search import VectorSearchService
from app.models.retrieval import RetrievalResponse # Import the response model


# Mark all tests in this module as async using pytest-asyncio
pytestmark = pytest.mark.asyncio

async def test_embed_query(mock_embedding_model: MagicMock):
    """Unit test for the _embed_query method."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=AsyncMock(), # Don't need chroma for this test
        top_k=3
    )
    query = "test query"
    embedding = service._embed_query(query)

    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    assert isinstance(embedding, list)
    assert len(embedding) == 384 # Check dimension based on mock encode output


async def test_search_found(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for the search method when documents are found."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    query = "find stuff"
    # Ensure mock_embed_query returns the embedding that triggers results in mock_chroma_collection.query
    # Our mock _embed_query returns [0.0, 1.0, ..., 383.0]
    # Our mock collection.query returns results if embedding[0] is 0.0

    results = await service.search(query)

    # Assertions
    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    # Use ANY for the embedding argument as generating the exact mock list is tedious
    mock_chroma_collection.query.assert_awaited_once_with(
        query_embeddings=[ANY], # Check that it was called with A list containing ONE embedding
        n_results=3,
        include=['documents']
    )
    assert results == ['mock chunk 1', 'mock chunk 2'] # Based on mock_chroma_collection setup


async def test_search_not_found(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for the search method when no documents are found."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    query = "find nothing"
    # Make encode return something that leads to no results in the mock query
    mock_embedding_model.encode.return_value.tolist.return_value = [[1.0] * 384] # Simulate different embedding

    results = await service.search(query)

    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    mock_chroma_collection.query.assert_awaited_once() # Check it was awaited
    assert results == [] # Expect empty list


async def test_search_chroma_error(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for the search method when ChromaDB query raises an error."""
    # Configure the mock query to raise an exception
    mock_chroma_collection.query.side_effect = Exception("Simulated ChromaDB connection error")

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    query = "cause error"

    with pytest.raises(HTTPException) as exc_info:
        await service.search(query)

    assert exc_info.value.status_code == 500
    assert "Failed to query vector database" in exc_info.value.detail
    mock_chroma_collection.query.assert_awaited_once() # Ensure it was called