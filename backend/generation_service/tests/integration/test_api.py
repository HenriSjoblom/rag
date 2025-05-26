"""
Integration tests for generation service API endpoints.

These tests verify that the API endpoints work correctly with real service
components while mocking external dependencies like LLM providers.
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
        assert data["service"] == "generation"

    def test_health_check_multiple_calls(self, integration_test_client):
        """Test health check endpoint handles multiple calls."""
        for _ in range(5):
            response = integration_test_client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["status"] == "ok"


class TestGenerationEndpointIntegration:
    """Integration tests for generation endpoint."""

    def test_generate_basic_request(self, integration_test_client):
        """Test basic generation request with valid input."""
        request_data = {
            "query": "What is the main purpose of this device?",
            "context_chunks": [
                "The iPhone is a smartphone device designed for communication and productivity.",
                "It features advanced camera technology and mobile applications.",
            ],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_generate_with_empty_context(self, integration_test_client):
        """Test generation with empty context chunks."""
        request_data = {"query": "Tell me about safety features", "context_chunks": []}

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_generate_with_single_context_chunk(self, integration_test_client):
        """Test generation with single context chunk."""
        request_data = {
            "query": "What safety information is provided?",
            "context_chunks": [
                "Important safety information: Keep device away from water and extreme temperatures."
            ],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_generate_with_multiple_context_chunks(self, integration_test_client):
        """Test generation with multiple context chunks."""
        request_data = {
            "query": "How do I use the camera?",
            "context_chunks": [
                "Camera features include portrait mode and night mode photography.",
                "To access camera, tap the Camera app icon on the home screen.",
                "Use volume buttons to take photos or touch the shutter button.",
            ],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_generate_with_large_context(
        self, integration_test_client, large_context_data
    ):
        """Test generation with large context data."""
        request_data = {
            "query": "What are the key features mentioned?",
            "context_chunks": large_context_data,
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_generate_with_special_characters(
        self, integration_test_client, special_character_data
    ):
        """Test generation with special characters in query and context."""
        request_data = {
            "query": special_character_data["query"],
            "context_chunks": special_character_data["context_chunks"],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_generate_with_unicode_content(self, integration_test_client):
        """Test generation with unicode characters."""
        request_data = {
            "query": "¿Cómo uso el dispositivo? 中文查询 русский запрос",
            "context_chunks": [
                "Información de seguridad: Mantenga el dispositivo alejado del agua.",
                "安全信息：设备应远离水和极端温度。",
                "Информация о безопасности: Берегите устройство от воды.",
            ],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_generate_with_long_query(self, integration_test_client):
        """Test generation with a long but valid query."""
        long_query = (
            "Can you provide detailed step-by-step instructions for setting up the device, "
            "including initial configuration, network setup, account creation, security settings, "
            "privacy configurations, and any important safety considerations that users should "
            "be aware of during the setup process? " * 10
        )
        request_data = {
            "query": long_query,
            "context_chunks": [
                "Setup instructions: First, power on the device and follow the on-screen prompts."
            ],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data

    def test_generate_response_structure(self, integration_test_client):
        """Test that generation response has the correct structure."""
        request_data = {
            "query": "What is this about?",
            "context_chunks": ["This is a user manual for a device."],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check required fields
        assert "answer" in data

        # Check field types
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_generate_multiple_concurrent_requests(self, integration_test_client):
        """Test handling multiple generation requests."""
        requests_data = [
            {
                "query": f"Question {i}: What features are mentioned?",
                "context_chunks": [
                    f"Feature {i}: Advanced functionality for user convenience."
                ],
            }
            for i in range(5)
        ]

        responses = []
        for request_data in requests_data:
            response = integration_test_client.post(
                "/api/v1/generate", json=request_data
            )
            responses.append(response)

        # Verify all requests succeeded
        for i, response in enumerate(responses):
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "answer" in data
            assert isinstance(data["answer"], str)


class TestGenerationEndpointValidation:
    """Integration tests for request validation and error handling."""

    def test_generate_with_empty_query(self, integration_test_client):
        """Test generation fails with empty query."""
        request_data = {"query": "", "context_chunks": ["Some context"]}

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_generate_with_whitespace_query(self, integration_test_client):
        """Test generation fails with whitespace-only query."""
        request_data = {"query": "   \n\t   ", "context_chunks": ["Some context"]}

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_generate_with_malformed_request(self, integration_test_client):
        """Test generation with malformed request data."""
        malformed_requests = [
            {},  # Missing required fields
            {"query": "test"},  # Missing context_chunks
            {"context_chunks": ["test"]},  # Missing query
            {"query": None, "context_chunks": ["test"]},  # Null query
            {"query": "test", "context_chunks": None},  # Null context_chunks
            {"query": 123, "context_chunks": ["test"]},  # Wrong query type
            {
                "query": "test",
                "context_chunks": "not a list",
            },  # Wrong context_chunks type
            {"query": "test", "context_chunks": [123]},  # Wrong context chunk type
        ]

        for request_data in malformed_requests:
            response = integration_test_client.post(
                "/api/v1/generate", json=request_data
            )
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_generate_with_invalid_json(self, integration_test_client):
        """Test generation with invalid JSON."""
        response = integration_test_client.post(
            "/api/v1/generate",
            data="invalid json{",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_generate_endpoint_methods(self, integration_test_client):
        """Test that generation endpoint only accepts POST requests."""
        request_data = {"query": "test query", "context_chunks": ["test context"]}

        # POST should work
        response = integration_test_client.post("/api/v1/generate", json=request_data)
        assert response.status_code == status.HTTP_200_OK

        # Other methods should fail
        response = integration_test_client.get("/api/v1/generate")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

        response = integration_test_client.put("/api/v1/generate", json=request_data)
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

        response = integration_test_client.delete("/api/v1/generate")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


class TestGenerationEndpointEdgeCases:
    """Integration tests for edge cases and error scenarios."""

    def test_generate_with_service_dependency_error(
        self, integration_test_client, mocker
    ):
        """Test generation when service dependency fails."""

        # Mock the dependency to raise an exception
        def failing_service(*args, **kwargs):
            raise RuntimeError("Service initialization failed")

        from app.deps import get_generation_service
        from app.main import app

        app.dependency_overrides[get_generation_service] = failing_service

        try:
            request_data = {"query": "test query", "context_chunks": ["test context"]}
            response = integration_test_client.post(
                "/api/v1/generate", json=request_data
            )

            # Should return unprocessable entity due to dependency failure
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    def test_generate_preserves_query_content(self, integration_test_client):
        """Test that generation preserves original query characteristics."""
        test_cases = [
            {"query": "Simple query", "context_chunks": ["Simple context"]},
            {
                "query": "Query with    extra    spaces",
                "context_chunks": ["Context with spaces"],
            },
            {
                "query": "Query\nwith\nnewlines",
                "context_chunks": ["Context\nwith\nnewlines"],
            },
            {"query": "Query\twith\ttabs", "context_chunks": ["Context\twith\ttabs"]},
        ]

        for test_case in test_cases:
            response = integration_test_client.post("/api/v1/generate", json=test_case)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "answer" in data
            assert isinstance(data["answer"], str)

    def test_generate_with_mixed_content_types(self, integration_test_client):
        """Test generation with mixed content in context chunks."""
        request_data = {
            "query": "What information is provided?",
            "context_chunks": [
                "",  # Empty string
                "Normal text content",
                "Content with numbers: 123, 456.789",
                "Content with symbols: @#$%^&*()",
                "Very short",
                "A" * 1000,  # Very long string
            ],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data

    def test_generate_with_empty_string_context_chunks(self, integration_test_client):
        """Test generation with context chunks containing empty strings."""
        request_data = {
            "query": "What can you tell me?",
            "context_chunks": ["", "", ""],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data

    def test_generate_answer_is_non_empty_string(self, integration_test_client):
        """Test that generated answers are always non-empty strings."""
        request_data = {
            "query": "Provide any information available",
            "context_chunks": ["Device manual information and usage guidelines."],
        }

        response = integration_test_client.post("/api/v1/generate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert data["answer"].strip()  # Should not be just whitespace

    def test_generate_content_type_validation(self, integration_test_client):
        """Test that endpoint requires correct content type."""
        request_data = {"query": "test query", "context_chunks": ["test context"]}

        # Should work with application/json
        response = integration_test_client.post(
            "/api/v1/generate",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == status.HTTP_200_OK

        # Should fail with wrong content type
        import json

        response = integration_test_client.post(
            "/api/v1/generate",
            data=json.dumps(request_data),
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGenerationEndpointPerformance:
    """Integration tests for performance and resource usage."""

    def test_generate_response_time_reasonable(self, integration_test_client):
        """Test that generation responses are returned in reasonable time."""
        import time

        request_data = {
            "query": "What safety features are mentioned?",
            "context_chunks": [
                "Safety features include automatic emergency calling and medical ID access.",
                "The device includes fall detection and emergency SOS functionality.",
            ],
        }

        start_time = time.time()
        response = integration_test_client.post("/api/v1/generate", json=request_data)
        end_time = time.time()

        assert response.status_code == status.HTTP_200_OK

        # Response should be reasonably fast (under 30 seconds for integration test)
        response_time = end_time - start_time
        assert response_time < 30.0, (
            f"Response took too long: {response_time:.2f} seconds"
        )

    def test_generate_handles_rapid_requests(self, integration_test_client):
        """Test that service handles rapid successive requests."""
        request_data = {
            "query": "Quick question about features",
            "context_chunks": ["Device features and capabilities"],
        }

        # Send 10 rapid requests
        responses = []
        for i in range(10):
            response = integration_test_client.post(
                "/api/v1/generate", json=request_data
            )
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "answer" in data

    def test_generate_with_varying_context_sizes(self, integration_test_client):
        """Test generation performance with varying context sizes."""
        base_query = "What information is provided about this topic?"

        # Test with different context sizes
        context_sizes = [1, 5, 10, 20]

        for size in context_sizes:
            context_chunks = [
                f"Context chunk {i} with relevant information." for i in range(size)
            ]
            request_data = {"query": base_query, "context_chunks": context_chunks}

            response = integration_test_client.post(
                "/api/v1/generate", json=request_data
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "answer" in data
            assert isinstance(data["answer"], str)
