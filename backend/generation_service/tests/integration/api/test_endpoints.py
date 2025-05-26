"""
Integration tests for generation service API endpoints.
"""

from fastapi import status


class TestGenerationEndpoint:
    """Test cases for the /generate endpoint."""

    def test_generate_endpoint_success(self, test_client_with_mocks):
        """Test successful generation request."""
        client, mock_service = test_client_with_mocks

        request_data = {
            "query": "What is FastAPI?",
            "context_chunks": ["FastAPI is a modern web framework."],
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Mocked LLM response for testing"

        # Verify service was called correctly
        mock_service.generate_answer.assert_awaited_once()
        call_args = mock_service.generate_answer.call_args[0][0]
        assert call_args.query == request_data["query"]
        assert call_args.context_chunks == request_data["context_chunks"]

    def test_generate_endpoint_empty_query(self, test_client_with_mocks):
        """Test generation request with empty query."""
        client, _ = test_client_with_mocks
        request_data = {"query": "", "context_chunks": ["Some context"]}

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_generate_endpoint_missing_query(self, test_client_with_mocks):
        """Test generation request with missing query field."""
        client, _ = test_client_with_mocks

        request_data = {"context_chunks": ["Some context"]}

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_generate_endpoint_empty_context(self, test_client_with_mocks):
        """Test generation request with empty context chunks."""
        client, mock_service = test_client_with_mocks

        request_data = {"query": "What is FastAPI?", "context_chunks": []}

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data

    def test_generate_endpoint_service_unavailable(self, test_client_with_mocks):
        """Test generation request when service is unavailable."""
        client, mock_service = test_client_with_mocks

        # Mock service to raise HTTPException
        from fastapi import HTTPException

        mock_service.generate_answer.side_effect = HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is temporarily unavailable",
        )

        request_data = {"query": "What is FastAPI?", "context_chunks": ["Some context"]}

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "LLM service is temporarily unavailable" in data["detail"]

    def test_generate_endpoint_internal_error(self, test_client_with_mocks):
        """Test generation request with internal service error."""
        client, mock_service = test_client_with_mocks

        # Mock service to raise generic exception
        mock_service.generate_answer.side_effect = Exception("Unexpected error")

        request_data = {"query": "What is FastAPI?", "context_chunks": ["Some context"]}

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "internal error occurred" in data["detail"].lower()

    def test_generate_endpoint_large_context(self, test_client_with_mocks):
        """Test generation request with large context chunks."""
        client, mock_service = test_client_with_mocks

        # Create large context chunks
        large_chunks = [
            f"This is chunk {i} with some content. " * 50 for i in range(10)
        ]

        request_data = {
            "query": "Summarize the information.",
            "context_chunks": large_chunks,
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data

    def test_generate_endpoint_special_characters(self, test_client_with_mocks):
        """Test generation request with special characters."""
        client, mock_service = test_client_with_mocks

        request_data = {
            "query": "What about émojis 🚀 and special chars: @#$%^&*()?",
            "context_chunks": ["Context with émojis 😊 and symbols: €£¥"],
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data


class TestAPIErrorHandling:
    """Test cases for API error handling."""

    def test_invalid_json_request(self, test_client_with_mocks):
        """Test request with invalid JSON."""
        client, _ = test_client_with_mocks

        response = client.post(
            "/api/v1/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_wrong_content_type(self, test_client_with_mocks):
        """Test request with wrong content type."""
        client, _ = test_client_with_mocks

        response = client.post(
            "/api/v1/generate",
            data="query=test&context_chunks=context",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        # Should still work as FastAPI can handle form data
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_200_OK,
        ]

    def test_missing_content_type(self, test_client_with_mocks):
        """Test request with missing content type."""
        client, _ = test_client_with_mocks

        response = client.post(
            "/api/v1/generate", data='{"query": "test", "context_chunks": []}'
        )

        # FastAPI should handle this gracefully
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_200_OK,
        ]


class TestResponseFormat:
    """Test cases for response format validation."""

    def test_response_structure(self, test_client_with_mocks):
        """Test that response follows expected structure."""
        client, _ = test_client_with_mocks

        request_data = {
            "query": "What is FastAPI?",
            "context_chunks": ["FastAPI context"],
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert isinstance(data, dict)
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_response_headers(self, test_client_with_mocks):
        """Test response headers."""
        client, _ = test_client_with_mocks

        request_data = {
            "query": "What is FastAPI?",
            "context_chunks": ["FastAPI context"],
        }

        response = client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json"


class TestEndpointPerformance:
    """Test cases for endpoint performance characteristics."""

    def test_concurrent_requests(self, test_client_with_mocks):
        """Test handling of concurrent requests."""
        client, mock_service = test_client_with_mocks

        # Mock async behavior
        async def mock_generate_answer(request):
            return f"Response for: {request.query}"

        mock_service.generate_answer.side_effect = mock_generate_answer

        request_data = {
            "query": "Test concurrent request",
            "context_chunks": ["Context"],
        }

        # Send multiple requests
        responses = []
        for i in range(5):
            response = client.post("/api/v1/generate", json=request_data)
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "answer" in data
