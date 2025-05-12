import pytest
from app.config import Settings
from app.models import AddDataResponse, RetrievalResponse
from fastapi import status
from fastapi.testclient import TestClient

from .conftest import get_injected_settings, get_docker_settings_config, is_docker_chromadb_available

# --- Tests for /retrieve endpoint ---


def test_retrieve_success_apples(client: TestClient, override_settings: Settings):
    """Test successful retrieval using the API endpoint."""

    settings_used_by_client_app = get_injected_settings(client)
    assert settings_used_by_client_app is override_settings

    print("Testing retrieval for apples...")
    request_data = {"query": "Tell me about apples"}
    response = client.post("/api/v1/retrieve", json=request_data)
    print(f"Response: {response.json()}")  # Debugging output
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Validate response format using the Pydantic model
    try:
        RetrievalResponse(**data)
    except Exception as e:
        pytest.fail(f"Response validation failed: {e}\nResponse data: {data}")

    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) > 0
    # Check if apple-related documents are likely present (order might vary)
    assert any("apples" in chunk.lower() for chunk in data["chunks"])
    # Check that we don't get ONLY orange results
    assert not all(
        "oranges" in chunk.lower() and "apples" not in chunk.lower()
        for chunk in data["chunks"]
    )


def test_retrieve_success_oranges(client: TestClient):
    """Test successful retrieval for a different query."""
    request_data = {"query": "What about oranges?"}
    response = client.post("/api/v1/retrieve", json=request_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    RetrievalResponse(**data)  # Validate schema
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) > 0
    assert any("oranges" in chunk.lower() for chunk in data["chunks"])


def test_retrieve_no_results(client: TestClient):
    """Test retrieval when the query doesn't match anything."""
    request_data = {"query": "information about bananas"}  # Bananas not in test data
    response = client.post("/api/v1/retrieve", json=request_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    RetrievalResponse(**data)  # Validate schema
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) == 0  # Expect empty list


def test_retrieve_invalid_input_no_query(client: TestClient):
    """Test API response when the 'query' field is missing."""
    request_data = {"message": "this is not the query field"}
    response = client.post("/api/v1/retrieve", json=request_data)

    # FastAPI/Pydantic validation should catch this
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_retrieve_invalid_input_empty_query(client: TestClient):
    """Test API response when the 'query' field is empty."""
    request_data = {
        "query": ""
    }  # Empty string might fail Pydantic validation if min_length=1
    response = client.post("/api/v1/retrieve", json=request_data)

    # FastAPI/Pydantic validation should catch this due to min_length=1
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# --- Tests for /add endpoint ---


def test_add_documents_success_and_retrieve(
    client: TestClient, override_settings: Settings
):
    """Test adding documents successfully and then retrieving them."""
    # Add new documents
    add_request_data = {
        "documents": {
            "doc_id_grapes": "Grapes are a type of fruit that grow in clusters.",
            "doc_id_vine": "Grapes grow on vines.",
        }
    }
    add_response = client.post("/api/v1/add", json=add_request_data)

    # Assert add success
    assert add_response.status_code == status.HTTP_200_OK
    add_data = add_response.json()
    try:
        AddDataResponse(**add_data)  # Validate response schema
    except Exception as e:
        pytest.fail(f"Add response validation failed: {e}\nResponse data: {add_data}")
    assert add_data["added_count"] == len(add_request_data["documents"])
    assert add_data["collection_name"] == override_settings.CHROMA_COLLECTION_NAME

    # Retrieve the newly added document
    retrieve_request_data = {"query": "Tell me about grapes"}
    retrieve_response = client.post("/api/v1/retrieve", json=retrieve_request_data)

    # Assert retrieval success
    assert retrieve_response.status_code == status.HTTP_200_OK
    retrieve_data = retrieve_response.json()
    RetrievalResponse(**retrieve_data)  # Validate schema
    assert isinstance(retrieve_data["chunks"], list)
    assert len(retrieve_data["chunks"]) > 0
    # Check if grape-related documents are now present
    assert any("grapes" in chunk.lower() for chunk in retrieve_data["chunks"])


def test_add_documents_empty_input(client: TestClient):
    """Test the /add endpoint with an empty documents dictionary."""
    request_data = {"documents": {}}
    response = client.post("/api/v1/add", json=request_data)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "No documents provided in the request."


def test_add_documents_invalid_input_format(client: TestClient):
    """Test the /add endpoint when 'documents' is not a dictionary."""
    request_data = {"documents": ["list", "not", "a", "dictionary"]}
    response = client.post("/api/v1/add", json=request_data)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_add_documents_missing_field(client: TestClient):
    """Test the /add endpoint when the 'documents' field is missing."""
    request_data = {"other_field": "some value"}
    response = client.post("/api/v1/add", json=request_data)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.skipif(
    not is_docker_chromadb_available(),
    reason="Docker-based ChromaDB is not available.",
)
def test_docker_add_documents_success(docker_client: TestClient, docker_settings: Settings):
    """Test adding documents successfully to Docker-based ChromaDB."""

    docker_settings: Settings = get_docker_settings_config()
    add_request_data = {
        "documents": {
            "doc_id_grapes": "Grapes are a type of fruit that grow in clusters.",
            "doc_id_vine": "Grapes grow on vines.",
        }
    }
    response = docker_client.post("/api/v1/add", json=add_request_data)

    if response.status_code != status.HTTP_200_OK:
        pytest.fail(
            f"Unexpected status code: {response.status_code}. Response: {response.json()}"
        )

    data = response.json()
    print(f"Add response data: {data}")  # Debugging output
    assert data["added_count"] == len(add_request_data["documents"])

    #assert data["collection_name"] == docker_settings.CHROMA_COLLECTION_NAME


@pytest.mark.skipif(
    not is_docker_chromadb_available(),
    reason="Docker-based ChromaDB is not available.",
)
def test_docker_add_documents_empty_input(
    docker_client: TestClient,
):
    """Test the /add endpoint with an empty documents dictionary in Docker-based ChromaDB."""
    request_data = {"documents": {}}
    print(f"Request data: {request_data}")  # Debugging output
    response = docker_client.post("/api/v1/add", json=request_data)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "No documents provided in the request."
