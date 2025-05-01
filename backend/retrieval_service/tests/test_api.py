import pytest
from fastapi import status
from fastapi.testclient import TestClient
import chromadb

from app.models import RetrievalResponse

# Mark all tests in this module as needing the session-scoped populated collection
pytestmark = [
    pytest.mark.usefixtures("populated_chroma_collection"),
    pytest.mark.asyncio # Needed if test functions themselves are async, though TestClient is sync
]


def test_retrieve_success_apples(client: TestClient):
    """Test successful retrieval using the API endpoint."""
    request_data = {"query": "Tell me about apples"}
    response = client.post("/api/v1/retrieve", json=request_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Validate response format using the Pydantic model
    try:
        RetrievalResponse(**data)
    except Exception as e:
        pytest.fail(f"Response validation failed: {e}\nResponse data: {data}")

    assert isinstance(data['chunks'], list)
    assert len(data['chunks']) > 0
    # Check if apple-related documents are likely present (order might vary)
    assert any("apples" in chunk.lower() for chunk in data['chunks'])
    # Check that we don't get ONLY orange results
    assert not all("oranges" in chunk.lower() and "apples" not in chunk.lower() for chunk in data['chunks'])


def test_retrieve_success_oranges(client: TestClient):
    """Test successful retrieval for a different query."""
    request_data = {"query": "What about oranges?"}
    response = client.post("/api/v1/retrieve", json=request_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    RetrievalResponse(**data) # Validate schema
    assert isinstance(data['chunks'], list)
    assert len(data['chunks']) > 0
    assert any("oranges" in chunk.lower() for chunk in data['chunks'])


def test_retrieve_no_results(client: TestClient):
    """Test retrieval when the query doesn't match anything."""
    request_data = {"query": "information about bananas"} # Bananas not in test data
    response = client.post("/api/v1/retrieve", json=request_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    RetrievalResponse(**data) # Validate schema
    assert isinstance(data['chunks'], list)
    assert len(data['chunks']) == 0 # Expect empty list


def test_retrieve_invalid_input_no_query(client: TestClient):
    """Test API response when the 'query' field is missing."""
    request_data = {"message": "this is not the query field"}
    response = client.post("/api/v1/retrieve", json=request_data)

    # FastAPI/Pydantic validation should catch this
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_retrieve_invalid_input_empty_query(client: TestClient):
    """Test API response when the 'query' field is empty."""
    request_data = {"query": ""} # Empty string might fail Pydantic validation if min_length=1
    response = client.post("/api/v1/retrieve", json=request_data)

    # FastAPI/Pydantic validation should catch this due to min_length=1
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY