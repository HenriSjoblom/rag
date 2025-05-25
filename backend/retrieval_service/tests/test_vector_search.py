from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from app.services.vector_search import VectorSearchService, lifespan_retrieval_service
from fastapi import HTTPException

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio


async def test_embed_query(mock_embedding_model: MagicMock):
    """Unit test for the _embed_query method."""
    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=AsyncMock(),
        collection_name="test_collection",
        top_k=3,
    )
    query = "test query"
    embedding_list = service._embed_query(query)

    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    assert isinstance(embedding_list, list)
    assert len(embedding_list) == 384


async def test_search_found(
    mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock
):
    """Unit test for the search method when documents are found."""
    mock_chroma_client = AsyncMock()
    mock_chroma_client.get_collection.return_value = mock_chroma_collection

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=mock_chroma_client,
        collection_name="test_collection",
        top_k=3,
    )
    query = "find stuff"

    results = await service.search(query)

    # Assertions
    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    mock_chroma_client.get_collection.assert_called_once_with(name="test_collection")
    # Use ANY for the embedding argument as generating the exact mock list is tedious
    mock_chroma_collection.query.assert_called_once_with(
        query_embeddings=[
            ANY
        ],  # Check that it was called with A list containing ONE embedding
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    assert results == [
        {
            "id": "doc_id_1",
            "text": "mock chunk 1",
            "metadata": {"source": "mock"},
            "distance": 0.1,
        },
        {
            "id": "doc_id_2",
            "text": "mock chunk 2",
            "metadata": {"source": "mock"},
            "distance": 0.2,
        },
    ]


async def test_search_not_found(
    mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock
):
    """Unit test for the search method when no documents are found."""
    mock_chroma_client = AsyncMock()
    mock_chroma_client.get_collection.return_value = mock_chroma_collection

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=mock_chroma_client,
        collection_name="test_collection",
        top_k=3,
    )
    query = "find nothing"

    # Simulate ChromaDB returning an empty result set
    empty_chroma_result = {
        "ids": [[]],
        "embeddings": None,
        "documents": [[]],  # Empty inner list indicates no documents found
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_chroma_collection.query.return_value = empty_chroma_result

    results = await service.search(query)

    mock_embedding_model.encode.assert_called_once_with(query, convert_to_numpy=True)
    mock_chroma_collection.query.assert_called_once()  # Check it was awaited
    # The service should process {'documents': [[]]} and return []
    assert results == []


async def test_search_chroma_error(
    mock_embedding_model: MagicMock, mock_chroma_collection: AsyncMock
):
    """Unit test for the search method when ChromaDB query raises an error."""
    # Configure the mock query to raise an exception
    mock_chroma_collection.query.side_effect = Exception(
        "Simulated ChromaDB connection error"
    )

    mock_chroma_client = AsyncMock()
    mock_chroma_client.get_collection.return_value = mock_chroma_collection

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=mock_chroma_client,
        collection_name="test_collection",
        top_k=3,
    )
    query = "cause error"

    with pytest.raises(HTTPException) as exc_info:
        await service.search(query)

    assert exc_info.value.status_code == 500
    assert "Failed to query vector database" in exc_info.value.detail
    mock_chroma_collection.query.assert_called_once()  # Ensure it was called


async def test_add_documents_success(mock_embedding_model: MagicMock):
    """Unit test for the add_documents method for successful addition."""
    mock_chroma_collection = AsyncMock()
    mock_chroma_client = AsyncMock()
    mock_chroma_client.get_collection.return_value = mock_chroma_collection

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=mock_chroma_client,
        collection_name="test_collection",
        top_k=3,
    )
    test_docs = {"id1": "text one", "id2": "text two"}
    doc_texts = list(test_docs.values())
    doc_ids = list(test_docs.keys())

    added_count = await service.add_documents(test_docs)

    # Check that encode was called with the correct texts
    mock_embedding_model.encode.assert_called_once_with(
        doc_texts, convert_to_tensor=False
    )
    # Check that chroma_collection.add was called with correct arguments
    mock_chroma_collection.add.assert_called_once_with(
        ids=doc_ids, documents=doc_texts, embeddings=ANY
    )
    assert added_count == len(test_docs)


async def test_add_documents_empty():
    """Unit test for add_documents with an empty input dictionary."""
    mock_embedding_model = MagicMock()
    mock_chroma_client = AsyncMock()

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=mock_chroma_client,
        collection_name="test_collection",
        top_k=3,
    )
    added_count = await service.add_documents({})

    assert added_count == 0
    mock_embedding_model.encode.assert_not_called()
    mock_chroma_client.get_collection.assert_not_called()


async def test_add_documents_embedding_error(mock_embedding_model: MagicMock):
    """Unit test for add_documents when embedding fails."""
    mock_chroma_client = AsyncMock()

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=mock_chroma_client,
        collection_name="test_collection",
        top_k=3,
    )
    test_docs = {"id1": "text one"}
    # Configure encode to raise an error
    mock_embedding_model.encode.side_effect = Exception("Simulated embedding error")

    with pytest.raises(HTTPException) as exc_info:
        await service.add_documents(test_docs)

    assert exc_info.value.status_code == 500
    assert "Failed to add documents" in exc_info.value.detail
    mock_embedding_model.encode.assert_called_once_with(
        list(test_docs.values()), convert_to_tensor=False
    )
    mock_chroma_client.get_collection.assert_not_called()


async def test_add_documents_chroma_error(mock_embedding_model: MagicMock):
    """Unit test for add_documents when ChromaDB add fails."""
    mock_chroma_collection = AsyncMock()
    mock_chroma_collection.add.side_effect = Exception("Simulated ChromaDB add error")
    mock_chroma_client = AsyncMock()
    mock_chroma_client.get_collection.return_value = mock_chroma_collection

    service = VectorSearchService(
        embedding_model=mock_embedding_model,
        chroma_client=mock_chroma_client,
        collection_name="test_collection",
        top_k=3,
    )
    test_docs = {"id1": "text one"}
    doc_texts = list(test_docs.values())
    doc_ids = list(test_docs.keys())

    with pytest.raises(HTTPException) as exc_info:
        await service.add_documents(test_docs)

    assert exc_info.value.status_code == 500
    assert "Failed to add documents" in exc_info.value.detail
    mock_embedding_model.encode.assert_called_once_with(
        doc_texts, convert_to_tensor=False
    )
    # Check that chroma_collection.add was called
    mock_chroma_collection.add.assert_called_once_with(
        ids=doc_ids, documents=doc_texts, embeddings=ANY
    )


@pytest.mark.asyncio
async def test_lifespan_retrieval_service_local_missing_path():
    with (
        patch("app.services.vector_search._embedding_model", None),
        patch("app.services.vector_search._chroma_client", None),
        patch("app.services.vector_search._chroma_collection", None),
    ):
        with pytest.raises(ValueError, match="chroma_path is required for local mode."):
            async with lifespan_retrieval_service(
                app=None,
                model_name="all-MiniLM-L6-v2",
                chroma_mode="local",
                chroma_path=None,  # Missing path
                collection_name="test_collection",
            ):
                pass


@pytest.mark.asyncio
async def test_lifespan_retrieval_service_docker_missing_host():
    with (
        patch("app.services.vector_search._embedding_model", None),
        patch("app.services.vector_search._chroma_client", None),
        patch("app.services.vector_search._chroma_collection", None),
    ):
        with pytest.raises(
            ValueError, match="chroma_host is required for docker mode."
        ):
            async with lifespan_retrieval_service(
                app=None,
                model_name="all-MiniLM-L6-v2",
                chroma_mode="docker",
                chroma_host=None,  # Missing host
                collection_name="test_collection",
            ):
                pass
