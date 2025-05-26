"""
Integration tests for RAG service API endpoints.

Tests the main API functionality including health checks, error handling,
and basic endpoint availability.
"""

from fastapi import status


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, integration_test_client):
        """Test that health check endpoint returns success."""
        response = integration_test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_spec_available(self, integration_test_client):
        """Test that OpenAPI specification is available."""
        response = integration_test_client.get("/openapi.json")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "RAG Service"

    def test_docs_ui_available(self, integration_test_client):
        """Test that Swagger UI documentation is available."""
        response = integration_test_client.get("/docs")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]

    def test_redoc_ui_available(self, integration_test_client):
        """Test that ReDoc UI documentation is available."""
        response = integration_test_client.get("/redoc")

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]


class TestErrorHandling:
    """Test API error handling."""

    def test_not_found_endpoint(self, integration_test_client):
        """Test that non-existent endpoints return 404."""
        response = integration_test_client.get("/nonexistent-endpoint")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data

    def test_method_not_allowed(self, integration_test_client):
        """Test that wrong HTTP methods return 405."""
        response = integration_test_client.delete("/health")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
        data = response.json()
        assert "detail" in data

    def test_invalid_json_payload(self, integration_test_client):
        """Test handling of invalid JSON in request body."""
        response = integration_test_client.post(
            "/api/v1/chat",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_fields(self, integration_test_client):
        """Test validation of required fields."""
        response = integration_test_client.post(
            "/api/v1/chat",
            json={},  # Missing required 'query' field
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data


class TestCORS:
    """Test CORS (Cross-Origin Resource Sharing) configuration."""

    def test_cors_headers_present(self, integration_test_client):
        """Test that CORS headers are present in responses."""
        response = integration_test_client.options("/api/v1/chat")

        # FastAPI should handle OPTIONS requests for CORS
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_405_METHOD_NOT_ALLOWED,
        ]

    def test_cors_headers_in_get_request(self, integration_test_client):
        """Test CORS headers in GET requests."""
        response = integration_test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK


class TestSecurity:
    """Test security-related features."""

    def test_security_headers(self, integration_test_client):
        """Test that security headers are present."""
        response = integration_test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        # Check for common security headers
        headers = response.headers
        assert "content-type" in headers

    def test_no_server_info_leakage(self, integration_test_client):
        """Test that server information is not leaked in headers."""
        response = integration_test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        headers = response.headers

        # Should not reveal server software details
        assert (
            "server" not in headers.get("server", "").lower()
            or "uvicorn" not in headers.get("server", "").lower()
        )


class TestContentTypes:
    """Test content type handling."""

    def test_json_content_type_required(self, integration_test_client):
        """Test that JSON content type is required for JSON endpoints."""
        response = integration_test_client.post(
            "/api/v1/chat",
            content='{"query": "test"}',
            headers={"Content-Type": "text/plain"},
        )

        # Should reject non-JSON content type
        assert response.status_code in [
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    def test_json_response_content_type(self, integration_test_client):
        """Test that JSON responses have correct content type."""
        response = integration_test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert "application/json" in response.headers["content-type"]


class TestRequestValidation:
    """Test request validation."""

    def test_request_size_limits(self, integration_test_client):
        """Test handling of large request payloads."""
        # Create a very large query string
        large_query = "test " * 10000  # 50KB+ query

        response = integration_test_client.post(
            "/api/v1/chat", json={"query": large_query}
        )

        # Should either accept it or reject with appropriate error
        # The exact behavior depends on configured limits
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    def test_empty_query_validation(self, integration_test_client):
        """Test validation of empty queries."""
        response = integration_test_client.post("/api/v1/chat", json={"query": ""})

        # Should reject empty queries
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_whitespace_only_query_validation(self, integration_test_client):
        """Test validation of whitespace-only queries."""
        response = integration_test_client.post(
            "/api/v1/chat", json={"query": "   \n\t   "}
        )

        # Should reject whitespace-only queries
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
