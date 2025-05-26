"""
Integration tests for RAG service chat endpoint.

Tests the complete chat pipeline including retrieval and generation service integration.
"""

import pytest
from app.deps import get_http_client
from fastapi import status


class TestChatEndpoint:
    """Test chat endpoint integration."""

    def test_successful_chat_query(
        self, configured_generation_client, sample_chat_queries
    ):
        """Test successful chat query with mocked services."""
        query = sample_chat_queries[0]

        response = configured_generation_client.post(
            "/api/v1/chat", json={"message": query}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_chat_with_different_queries(
        self, configured_generation_client, sample_chat_queries
    ):
        """Test chat endpoint with various query types."""
        for query in sample_chat_queries:
            response = configured_generation_client.post(
                "/api/v1/chat", json={"message": query}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "response" in data
            assert isinstance(data["response"], str)

    def test_chat_response_structure(self, configured_generation_client):
        """Test that chat response has correct structure."""
        response = configured_generation_client.post(
            "/api/v1/chat", json={"message": "Test query"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check required fields
        assert "response" in data
        assert isinstance(data["response"], str)
        assert "query" in data
        assert isinstance(data["query"], str)

        # Optional fields that might be present
        if "metadata" in data:
            assert isinstance(data["metadata"], dict)
        if "sources" in data:
            assert isinstance(data["sources"], list)

    def test_chat_with_special_characters(self, configured_generation_client):
        """Test chat with special characters and unicode."""
        special_queries = [
            "What about émojis 🤔 and ñiño?",
            "Query with quotes: 'single' and \"double\"",
            "Math symbols: α + β = γ, ∑(x²)",
            "Programming: SELECT * FROM table WHERE id > 5;",
        ]

        for query in special_queries:
            response = configured_generation_client.post(
                "/api/v1/chat", json={"message": query}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "response" in data


class TestChatServiceIntegration:
    """Test integration with retrieval and generation services."""

    @pytest.mark.asyncio
    async def test_retrieval_service_called(
        self, integration_test_client, mock_retrieval_response, mocker
    ):
        """Test that retrieval service is called during chat."""
        # Configure mock to track calls
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        # Configure retrieval response
        retrieval_mock = mocker.MagicMock()
        retrieval_mock.status_code = 200
        retrieval_mock.is_success = True
        retrieval_mock.json.return_value = mock_retrieval_response
        retrieval_mock.raise_for_status.return_value = None

        # Configure generation response
        generation_mock = mocker.MagicMock()
        generation_mock.status_code = 200
        generation_mock.is_success = True
        generation_mock.json.return_value = {"answer": "Generated response"}
        generation_mock.raise_for_status.return_value = None

        async def request_side_effect(method, url, **kwargs):
            if "retrieve" in str(url):
                return retrieval_mock
            elif "generate" in str(url):
                return generation_mock
            return mocker.MagicMock()

        mock_http_client.request.side_effect = request_side_effect

        # Make chat request
        response = integration_test_client.post(
            "/api/v1/chat", json={"message": "Test query"}
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify retrieval service was called (check request method instead of post)
        assert mock_http_client.request.called

        # Check that retrieval URL was called
        calls = [
            call
            for call in mock_http_client.request.call_args_list
            if "retrieve" in str(call[0][1])
        ]  # Check URL in second argument
        assert len(calls) > 0

    def test_chat_pipeline_data_flow(self, integration_test_client, mocker):
        """Test that data flows correctly through the chat pipeline."""
        # Configure mock to verify data flow
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        retrieval_query = None
        generation_context = None
        generation_query = None

        async def request_side_effect(method, url, **kwargs):
            nonlocal retrieval_query, generation_context, generation_query
            mock_response = mocker.MagicMock()
            mock_response.status_code = 200
            mock_response.is_success = True
            mock_response.raise_for_status.return_value = None

            if "retrieve" in str(url):
                retrieval_query = kwargs.get("json", {}).get("query")
                mock_response.json.return_value = {
                    "chunks": ["Retrieved context 1", "Retrieved context 2"]
                }
            elif "generate" in str(url):
                generation_data = kwargs.get("json", {})
                generation_context = generation_data.get("context_chunks")
                generation_query = generation_data.get("query")
                mock_response.json.return_value = {
                    "answer": "Generated answer using context"
                }

            return mock_response

        mock_http_client.request.side_effect = request_side_effect

        # Make chat request
        test_query = "What are iPhone camera features?"
        response = integration_test_client.post(
            "/api/v1/chat", json={"message": test_query}
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify data flow
        assert retrieval_query == test_query
        assert generation_query == test_query
        assert generation_context is not None
        assert isinstance(generation_context, (list, str))


class TestChatErrorHandling:
    """Test chat endpoint error handling."""

    def test_retrieval_service_error(self, integration_test_client, mocker):
        """Test handling of retrieval service errors."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        async def post_side_effect(url, **kwargs):
            if "retrieve" in str(url):
                # Simulate retrieval service error
                error_response = mocker.MagicMock()
                error_response.status_code = 500
                error_response.is_success = False
                error_response.raise_for_status.side_effect = Exception(
                    "Retrieval service error"
                )
                return error_response
            else:
                # Normal response for other services
                mock_response = mocker.MagicMock()
                mock_response.status_code = 200
                mock_response.is_success = True
                mock_response.json.return_value = {"answer": "Generated response"}
                return mock_response

        mock_http_client.post.side_effect = post_side_effect

        response = integration_test_client.post(
            "/api/v1/chat", json={"message": "Test query"}
        )

        # Should return error when retrieval service fails
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data

    def test_generation_service_error(self, integration_test_client, mocker):
        """Test handling of generation service errors."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        async def post_side_effect(url, **kwargs):
            if "retrieve" in str(url):
                # Normal retrieval response
                mock_response = mocker.MagicMock()
                mock_response.status_code = 200
                mock_response.is_success = True
                mock_response.json.return_value = {
                    "chunks": ["context"],
                    "query": "test",
                }
                mock_response.raise_for_status.return_value = None
                return mock_response
            elif "generate" in str(url):
                # Simulate generation service error
                error_response = mocker.MagicMock()
                error_response.status_code = 500
                error_response.is_success = False
                error_response.raise_for_status.side_effect = Exception(
                    "Generation service error"
                )
                return error_response

        mock_http_client.post.side_effect = post_side_effect

        response = integration_test_client.post(
            "/api/v1/chat", json={"message": "Test query"}
        )

        # Should return error when generation service fails
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data

    def test_network_timeout_handling(self, integration_test_client, mocker):
        """Test handling of network timeouts."""
        mock_http_client = integration_test_client.app.dependency_overrides[
            get_http_client
        ]()

        async def post_side_effect(url, **kwargs):
            # Simulate network timeout
            raise Exception("Request timeout")

        mock_http_client.post.side_effect = post_side_effect

        response = integration_test_client.post(
            "/api/v1/chat", json={"message": "Test query"}
        )

        # Should handle timeout gracefully
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestChatPerformance:
    """Test chat endpoint performance characteristics."""

    def test_concurrent_chat_requests(self, configured_generation_client):
        """Test handling of concurrent chat requests."""
        import concurrent.futures

        def make_chat_request(query_num):
            return configured_generation_client.post(
                "/api/v1/chat", json={"message": f"Test query {query_num}"}
            )

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_chat_request, i) for i in range(5)]
            responses = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "response" in data
