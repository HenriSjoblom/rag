import pytest
from unittest.mock import MagicMock
# Import HTTPException and status for the side_effect
from fastapi import status, HTTPException
from fastapi.testclient import TestClient

from app.deps import get_generation_service
from app.models import GenerateResponse

pytestmark = pytest.mark.usefixtures("client")

# --- Test Data ---
VALID_REQUEST_DATA = {
    "query": "What is the capital of France?",
    "context_chunks": [
        "France is a country in Western Europe.",
        "Paris is the capital and largest city of France."
    ]
}

# --- Helper to get the mock service object injected via fixture ---
def get_injected_mock_service(client: TestClient) -> MagicMock:
     """Retrieves the mock GenerationService instance injected via dependency overrides."""
     # Access the mocked service injected into the app via the client's app instance
     mock_service = client.app.dependency_overrides.get(get_generation_service)
     if not mock_service:
         pytest.fail("Dependency override for get_generation_service not found in test app.")

     return mock_service()

# --- API Tests ---
def test_generate_success(client: TestClient):
    """Test the /generate endpoint for a successful response."""
    # Arrange
    expected_answer = "Based on the context, Paris is the capital of France."
    # Get the mock service instance that the API will use
    mock_service = get_injected_mock_service(client)
    # Configure the mock's generate_answer method for this test
    # Assumes the mock service has an async generate_answer method
    mock_service.generate_answer.return_value = expected_answer

    # Act
    response = client.post("/api/v1/generate", json=VALID_REQUEST_DATA)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    try:
        GenerateResponse(**data)
    except Exception as e:
        pytest.fail(f"Response validation failed: {e}\nResponse data: {data}")

    assert data["answer"] == expected_answer
    # Check that the mocked service's method was called
    mock_service.generate_answer.assert_awaited_once()


def test_generate_llm_api_error(client: TestClient):
    """Test the /generate endpoint when the downstream service call fails."""
    # Arrange
    # Simulate the specific HTTPException the service layer would raise
    error_detail = "Failed to get response from LLM: Simulated LLM Unavailable"
    simulated_error = HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=error_detail
    )
    mock_service = get_injected_mock_service(client)

    # Act
    response = client.post("/api/v1/generate", json=VALID_REQUEST_DATA)

    # Assert
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    # Check the detail message matches what the mock raised
    assert error_detail in response.json()["detail"]
    # Ensure the failing call was attempted
    mock_service.generate_answer.assert_awaited_once()


@pytest.mark.parametrize(
    "invalid_payload, expected_detail_part",
    [
        ({"context_chunks": ["chunk1"]}, "Field required"), # Missing query
        ({"query": "abc"}, "Field required"), # Missing context_chunks
        ({"query": "abc", "context_chunks": "not a list"}, "Input should be a valid list"),
        ({"query": "", "context_chunks": []}, "String should have at least 1 character"),
    ],
    ids=["missing_query", "missing_chunks", "wrong_type_chunks", "empty_query"]
)
def test_generate_invalid_input(client: TestClient, invalid_payload: dict, expected_detail_part: str):
    """Test the /generate endpoint with various invalid inputs."""

    print(f"Testing with invalid payload: {invalid_payload}")
    print(f"Expected detail part: {expected_detail_part}")
    response = client.post("/api/v1/generate", json=invalid_payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert expected_detail_part in str(response.json()["detail"])