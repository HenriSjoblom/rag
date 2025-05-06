import pytest
from unittest.mock import MagicMock, AsyncMock, ANY
from fastapi import HTTPException

from app.services.vector_search import VectorSearchService
from app.models import RetrievalResponse


# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

async def test_embed_query(mock_embedding_model: MagicMock):
    """Unit test for the _embed_query method."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=AsyncMock(),
        top_k=3
    )
    query = "test query"
    embedding_list = service._embed_query(query)

    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    assert isinstance(embedding_list, list)
    assert len(embedding_list) == 384


async def test_search_found(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for the search method when documents are found."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    query = "find stuff"

    results = await service.search(query)

    # Assertions
    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    # Use ANY for the embedding argument as generating the exact mock list is tedious
    mock_chroma_collection.query.assert_called_once_with(
        query_embeddings=[ANY], # Check that it was called with A list containing ONE embedding
        n_results=3,
        include=['documents', 'distances']
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

    # Simulate ChromaDB returning an empty result set
    empty_chroma_result = {
        'ids': [[]],
        'embeddings': None,
        'documents': [[]], # Empty inner list indicates no documents found
        'metadatas': [[]],
        'distances': [[]]
    }
    mock_chroma_collection.query.return_value = empty_chroma_result

    results = await service.search(query)

    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    mock_chroma_collection.query.assert_called_once() # Check it was awaited
    # The service should process {'documents': [[]]} and return []
    assert results == []


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
    mock_chroma_collection.query.assert_called_once() # Ensure it was called

async def test_add_documents_success(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for the add_documents method for successful addition."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    test_docs = {"id1": "text one", "id2": "text two"}
    doc_texts = list(test_docs.values())
    doc_ids = list(test_docs.keys())

    added_count = await service.add_documents(test_docs)

    # Check that encode was called with the correct texts
    mock_embedding_model.encode.assert_called_once_with(doc_texts, convert_to_tensor=False)
    # Check that chroma_collection.add was called with correct arguments
    mock_chroma_collection.add.assert_called_once_with(
        ids=doc_ids,
        documents=doc_texts,
        embeddings=ANY
    )
    assert added_count == len(test_docs)


async def test_add_documents_empty(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for add_documents with an empty input dictionary."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    added_count = await service.add_documents({})

    assert added_count == 0
    mock_embedding_model.encode.assert_not_called()
    mock_chroma_collection.add.assert_not_called()


async def test_add_documents_embedding_error(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for add_documents when embedding fails."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    test_docs = {"id1": "text one"}
    # Configure encode to raise an error
    mock_embedding_model.encode.side_effect = Exception("Simulated embedding error")

    with pytest.raises(HTTPException) as exc_info:
        await service.add_documents(test_docs)

    assert exc_info.value.status_code == 500
    assert "Failed to add documents" in exc_info.value.detail # Check detail from add_documents
    mock_embedding_model.encode.assert_called_once_with(list(test_docs.values()), convert_to_tensor=False)
    mock_chroma_collection.add.assert_not_called()


async def test_add_documents_chroma_error(mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock):
    """Unit test for add_documents when ChromaDB add fails."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_collection=mock_chroma_collection,
        top_k=3
    )
    test_docs = {"id1": "text one"}
    doc_texts = list(test_docs.values())
    doc_ids = list(test_docs.keys())

    # Configure chroma add to raise an error
    mock_chroma_collection.add.side_effect = Exception("Simulated ChromaDB add error")

    with pytest.raises(HTTPException) as exc_info:
        await service.add_documents(test_docs)

    assert exc_info.value.status_code == 500
    assert "Failed to add documents" in exc_info.value.detail
    mock_embedding_model.encode.assert_called_once_with(doc_texts, convert_to_tensor=False)
    # Check that chroma_collection.add was called
    mock_chroma_collection.add.assert_called_once_with(
        ids=doc_ids,
        documents=doc_texts,
        embeddings=ANY
    )