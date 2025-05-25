"""
Integration tests for retrieval service API endpoints.

These tests verify that the API endpoints work correctly with real service
components while mocking external dependencies like ChromaDB and embedding models.
"""

from fastapi import status


class TestHealthEndpointIntegration:
    """Integration tests for health endpoint."""

    def test_health_check_endpoint(self, integration_test_client):
        """Test health check endpoint returns correct response."""
        response = integration_test_client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "retrieval"

    def test_health_check_multiple_calls(self, integration_test_client):
        """Test health check endpoint handles multiple calls."""
        for _ in range(5):
            response = integration_test_client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["status"] == "ok"


class TestRetrievalEndpointIntegration:
    """Integration tests for retrieval endpoint."""

    def test_retrieve_with_empty_collection(self, integration_test_client):
        """Test retrieval when collection is empty."""
        request_data = {"query": "test query"}

        response = integration_test_client.post("/api/v1/retrieve", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "chunks" in data
        assert isinstance(data["chunks"], list)
        assert data["query"] == "test query"

    def test_retrieve_with_populated_collection(
        self, populated_integration_client, sample_retrieval_queries
    ):
        """Test retrieval with pre-populated collection data."""
        for query in sample_retrieval_queries[:3]:  # Test first 3 queries
            request_data = {"query": query}

            response = populated_integration_client.post(
                "/api/v1/retrieve", json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "chunks" in data
            assert isinstance(data["chunks"], list)
            assert data["query"] == query
            assert data.get("collection_name") is not None

    def test_retrieve_returns_relevant_content(self, populated_integration_client):
        """Test that retrieval returns content relevant to the query."""
        # Test specific query that should match our test documents
        request_data = {"query": "iPhone camera features"}

        response = populated_integration_client.post(
            "/api/v1/retrieve", json=request_data
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["chunks"]) > 0
        assert data["query"] == "iPhone camera features"

    def test_retrieve_with_long_query(self, integration_test_client):
        """Test retrieval with a long but valid query."""
        long_query = "smartphone features " * 100  # About 1800 characters
        request_data = {"query": long_query}

        response = integration_test_client.post("/api/v1/retrieve", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == long_query

    def test_retrieve_with_too_long_query(self, integration_test_client):
        """Test retrieval fails with query that exceeds length limit."""
        # Create query longer than 10,000 characters
        too_long_query = "a" * 10001
        request_data = {"query": too_long_query}

        response = integration_test_client.post("/api/v1/retrieve", json=request_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "too long" in data["detail"].lower()

    def test_retrieve_with_empty_query(self, integration_test_client):
        """Test retrieval fails with empty query."""
        request_data = {"query": ""}

        response = integration_test_client.post("/api/v1/retrieve", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_retrieve_with_whitespace_query(self, integration_test_client):
        """Test retrieval with whitespace-only query."""
        request_data = {"query": "   "}

        response = integration_test_client.post("/api/v1/retrieve", json=request_data)

        # Should be valid since it's not empty, just whitespace
        assert response.status_code == status.HTTP_200_OK

    def test_retrieve_with_special_characters(self, integration_test_client):
        """Test retrieval with special characters in query."""
        special_queries = [
            "query with @#$%^&* special chars",
            "query with émoji 📱",
            "query with newlines\nand\ttabs",
            "query with 'quotes' and \"double quotes\"",
        ]

        for query in special_queries:
            request_data = {"query": query}
            response = integration_test_client.post(
                "/api/v1/retrieve", json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["query"] == query

    def test_retrieve_with_unicode_query(self, integration_test_client):
        """Test retrieval with unicode characters."""
        unicode_queries = [
            "Recherche en français",
            "Búsqueda en español",
            "Поиск на русском",
            "中文搜索",
            "日本語検索",
        ]

        for query in unicode_queries:
            request_data = {"query": query}
            response = integration_test_client.post(
                "/api/v1/retrieve", json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["query"] == query

    def test_retrieve_response_structure(self, populated_integration_client):
        """Test that retrieval response has the correct structure."""
        request_data = {"query": "test query"}

        response = populated_integration_client.post(
            "/api/v1/retrieve", json=request_data
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check required fields
        assert "chunks" in data
        assert "collection_name" in data
        assert "query" in data

        # Check field types
        assert isinstance(data["chunks"], list)
        assert (
            isinstance(data["collection_name"], str) or data["collection_name"] is None
        )
        assert isinstance(data["query"], str) or data["query"] is None

    def test_retrieve_chunks_are_strings(self, populated_integration_client):
        """Test that returned chunks are valid strings."""
        request_data = {"query": "iPhone features"}

        response = populated_integration_client.post(
            "/api/v1/retrieve", json=request_data
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        for chunk in data["chunks"]:
            assert isinstance(chunk, str)
            assert len(chunk) > 0  # Should not be empty strings

    def test_retrieve_multiple_concurrent_requests(
        self, populated_integration_client, sample_retrieval_queries
    ):
        """Test handling multiple retrieval requests."""
        responses = []

        # Send multiple requests
        for query in sample_retrieval_queries:
            request_data = {"query": query}
            response = populated_integration_client.post(
                "/api/v1/retrieve", json=request_data
            )
            responses.append(response)

        # Verify all requests succeeded
        for i, response in enumerate(responses):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["query"] == sample_retrieval_queries[i]

    def test_retrieve_with_malformed_request(self, integration_test_client):
        """Test retrieval with malformed request data."""
        malformed_requests = [
            {},  # Missing query
            {"query": None},  # Null query
            {"wrong_field": "test"},  # Wrong field name
            {"query": 123},  # Wrong type
        ]

        for request_data in malformed_requests:
            response = integration_test_client.post(
                "/api/v1/retrieve", json=request_data
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_retrieve_with_invalid_json(self, integration_test_client):
        """Test retrieval with invalid JSON."""
        response = integration_test_client.post(
            "/api/v1/retrieve",
            data="invalid json{",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_retrieve_endpoint_methods(self, integration_test_client):
        """Test that retrieval endpoint only accepts POST requests."""
        request_data = {"query": "test"}

        # POST should work
        response = integration_test_client.post("/api/v1/retrieve", json=request_data)
        assert response.status_code == status.HTTP_200_OK

        # Other methods should fail
        response = integration_test_client.get("/api/v1/retrieve")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

        response = integration_test_client.put("/api/v1/retrieve", json=request_data)
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

        response = integration_test_client.delete("/api/v1/retrieve")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


class TestRetrievalEndpointEdgeCases:
    """Integration tests for edge cases and error scenarios."""

    def test_retrieve_with_service_dependency_error(
        self, integration_test_client, mocker
    ):
        """Test retrieval when service dependency fails."""

        # Mock the dependency to raise an exception
        def failing_service(*args, **kwargs):
            raise RuntimeError("Service initialization failed")

        from app.deps import get_vector_search_service
        from app.main import app

        app.dependency_overrides[get_vector_search_service] = failing_service

        try:
            request_data = {"query": "test query"}
            response = integration_test_client.post(
                "/api/v1/retrieve", json=request_data
            )

            # Should return unprocessable entity due to dependency failure
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    def test_retrieve_preserves_query_content(self, populated_integration_client):
        """Test that retrieval preserves original query content exactly."""
        test_queries = [
            "Simple query",
            "Query with    extra    spaces",
            "Query\nwith\nnewlines",
            "Query\twith\ttabs",
            "Query with trailing spaces   ",
            "   Query with leading spaces",
        ]

        for original_query in test_queries:
            request_data = {"query": original_query}
            response = populated_integration_client.post(
                "/api/v1/retrieve", json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["query"] == original_query  # Should preserve exactly

    def test_retrieve_collection_name_consistency(
        self, populated_integration_client, integration_settings
    ):
        """Test that collection name in response is consistent."""
        request_data = {"query": "test query"}

        # Make multiple requests
        collection_names = []
        for _ in range(3):
            response = populated_integration_client.post(
                "/api/v1/retrieve", json=request_data
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            collection_names.append(data.get("collection_name"))

        # All collection names should be the same
        assert len(set(collection_names)) == 1
        # Should match settings
        assert collection_names[0] == integration_settings.CHROMA_COLLECTION_NAME
