"""
Integration tests for GenerationService.

These tests verify that GenerationService works correctly with real
service components while mocking external dependencies like LLM providers.
"""

import pytest
from app.models import GenerateRequest
from app.services.generation import GenerationService
from fastapi import HTTPException


class TestGenerationServiceIntegration:
    """Integration tests for GenerationService with all dependencies."""

    def test_service_initialization_with_dependencies(
        self, integration_settings, mock_langchain_components
    ):
        """Test service initializes correctly with all dependencies."""
        service = GenerationService(settings=integration_settings)

        assert service.settings == integration_settings
        assert service.chat_model is not None
        assert service.prompt_template is not None
        assert service.output_parser is not None
        assert service.rag_chain is not None

    def test_service_uses_correct_settings(
        self, integration_generation_service, integration_settings
    ):
        """Test that service uses the correct settings."""
        assert (
            integration_generation_service.settings.LLM_PROVIDER
            == integration_settings.LLM_PROVIDER
        )
        assert (
            integration_generation_service.settings.LLM_MODEL_NAME
            == integration_settings.LLM_MODEL_NAME
        )
        assert (
            integration_generation_service.settings.LLM_TEMPERATURE
            == integration_settings.LLM_TEMPERATURE
        )
        assert (
            integration_generation_service.settings.LLM_MAX_TOKENS
            == integration_settings.LLM_MAX_TOKENS
        )

    @pytest.mark.asyncio
    async def test_generate_answer_with_context(self, integration_generation_service):
        """Test generate_answer with context chunks."""
        request = GenerateRequest(
            query="What is FastAPI?",
            context_chunks=["FastAPI is a modern web framework for Python."],
        )

        result = await integration_generation_service.generate_answer(request)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "FastAPI" in result or "query" in result

    @pytest.mark.asyncio
    async def test_generate_answer_without_context(
        self, integration_generation_service
    ):
        """Test generate_answer without context chunks."""
        request = GenerateRequest(query="What is machine learning?", context_chunks=[])

        result = await integration_generation_service.generate_answer(request)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_answer_with_multiple_contexts(
        self, integration_generation_service
    ):
        """Test generate_answer with multiple context chunks."""
        request = GenerateRequest(
            query="Explain the differences",
            context_chunks=[
                "Python is a high-level programming language.",
                "JavaScript is used for web development.",
                "Both are interpreted languages.",
            ],
        )

        result = await integration_generation_service.generate_answer(request)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_answer_with_long_query(
        self, integration_generation_service
    ):
        """Test generate_answer with very long query."""
        long_query = "This is a very long query. " * 100
        request = GenerateRequest(
            query=long_query, context_chunks=["Some context information."]
        )

        result = await integration_generation_service.generate_answer(request)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_answer_with_special_characters(
        self, integration_generation_service
    ):
        """Test generate_answer with special characters and unicode."""
        request = GenerateRequest(
            query="How to use émojis 🚀 and symbols: €£¥?",
            context_chunks=["Unicode support: 你好世界", "Emojis: 😊🎉"],
        )

        result = await integration_generation_service.generate_answer(request)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_answer_format_context_functionality(
        self, integration_generation_service
    ):
        """Test that _format_context works correctly."""
        # Test with multiple chunks
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        formatted = integration_generation_service._format_context(chunks)

        assert "Chunk 1" in formatted
        assert "Chunk 2" in formatted
        assert "Chunk 3" in formatted
        assert "---" in formatted  # Separator

        # Test with empty chunks
        empty_formatted = integration_generation_service._format_context([])
        assert "No context provided" in empty_formatted

    def test_service_health_check(self, integration_generation_service):
        """Test service health check functionality."""
        assert integration_generation_service.is_healthy() is True

    @pytest.mark.asyncio
    async def test_generate_answer_error_handling_chain_failure(
        self, integration_settings, mocker
    ):
        """Test error handling when RAG chain fails."""
        # Create service
        service = GenerationService(settings=integration_settings)

        # Mock the RAG chain to raise an exception
        mock_chain = mocker.AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("Chain execution failed")
        mocker.patch.object(service, "rag_chain", mock_chain)

        request = GenerateRequest(query="Test query", context_chunks=["Test context"])

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(request)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_generate_answer_error_handling_rate_limit(
        self, integration_settings, mocker
    ):
        """Test error handling for rate limit errors."""
        service = GenerationService(settings=integration_settings)

        # Mock the RAG chain to raise a rate limit exception
        mock_chain = mocker.AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("Rate limit exceeded")
        mocker.patch.object(service, "rag_chain", mock_chain)

        request = GenerateRequest(query="Test query", context_chunks=["Test context"])

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(request)
        assert exc_info.value.status_code == 503
        assert "rate limit" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_generate_answer_error_handling_authentication(
        self, integration_settings, mocker
    ):
        """Test error handling for authentication errors."""
        service = GenerationService(settings=integration_settings)

        # Mock the RAG chain to raise an authentication exception
        mock_chain = mocker.AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("Authentication failed")
        mocker.patch.object(service, "rag_chain", mock_chain)

        request = GenerateRequest(query="Test query", context_chunks=["Test context"])

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(request)
        assert exc_info.value.status_code == 503
        assert "authentication" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_generate_answer_error_handling_timeout(
        self, integration_settings, mocker
    ):
        """Test error handling for timeout errors."""
        service = GenerationService(settings=integration_settings)

        # Mock the RAG chain to raise a timeout exception
        mock_chain = mocker.AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("Request timeout")
        mocker.patch.object(service, "rag_chain", mock_chain)

        request = GenerateRequest(query="Test query", context_chunks=["Test context"])

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(request)
        assert exc_info.value.status_code == 503
        assert "timed out" in exc_info.value.detail.lower()


class TestGenerationServiceConfiguration:
    """Integration tests for GenerationService configuration."""

    def test_service_initialization_openai_provider(
        self, integration_test_data_dir, mocker
    ):
        """Test service initialization with OpenAI provider."""
        from app.config import Settings

        settings = Settings(
            LLM_PROVIDER="openai",
            LLM_MODEL_NAME="gpt-3.5-turbo",
            LLM_TEMPERATURE=0.5,
            LLM_MAX_TOKENS=300,
            OPENAI_API_KEY="test-key-integration",
            LOG_LEVEL="INFO",
        )

        # Mock LangChain components
        mock_model = mocker.MagicMock()
        mocker.patch("app.services.generation.ChatOpenAI", return_value=mock_model)
        mocker.patch("app.services.generation.ChatPromptTemplate")
        mocker.patch("app.services.generation.StrOutputParser")

        service = GenerationService(settings=settings)

        assert service.settings == settings
        assert service.chat_model is not None

    def test_service_initialization_invalid_provider(self, integration_test_data_dir):
        """Test service initialization with invalid provider."""
        from app.config import Settings

        with pytest.raises(ValueError):
            Settings(
                LLM_PROVIDER="invalid_provider",
                LLM_MODEL_NAME="test-model",
                OPENAI_API_KEY="test-key",
            )

    def test_service_initialization_missing_api_key(
        self, integration_test_data_dir, monkeypatch
    ):
        """Test service initialization with missing API key."""
        from typing import Optional

        from pydantic import Field, SecretStr, ValidationError, model_validator
        from pydantic_settings import BaseSettings, SettingsConfigDict

        # Clear any environment variables
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create isolated Settings class for testing
        class TestSettings(BaseSettings):
            LLM_PROVIDER: str = Field(default="openai")
            OPENAI_API_KEY: Optional[SecretStr] = Field(default=None)

            @model_validator(mode="after")
            def validate_openai_configuration(self) -> "TestSettings":
                if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
                    raise ValueError(
                        "OPENAI_API_KEY is required when using OpenAI provider"
                    )
                return self

            model_config = SettingsConfigDict(env_file=None, extra="ignore")

        with pytest.raises(ValidationError):
            TestSettings(
                LLM_PROVIDER="openai",
                # OPENAI_API_KEY not provided
            )


class TestGenerationServiceIntegrationEdgeCases:
    """Integration tests for edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_generate_answer_with_very_large_context(
        self, integration_generation_service
    ):
        """Test generate_answer with very large context chunks."""
        large_chunks = [f"Large context chunk {i}. " * 1000 for i in range(10)]

        request = GenerateRequest(
            query="Summarize the information", context_chunks=large_chunks
        )

        result = await integration_generation_service.generate_answer(request)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_answer_with_empty_strings(
        self, integration_generation_service
    ):
        """Test generate_answer with empty context strings."""
        request = GenerateRequest(
            query="Test query", context_chunks=["", "  ", "Valid content", ""]
        )

        result = await integration_generation_service.generate_answer(request)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_answer_concurrent_requests(
        self, integration_generation_service
    ):
        """Test handling multiple concurrent generation requests."""
        import asyncio

        requests = [
            GenerateRequest(
                query=f"Query {i}", context_chunks=[f"Context for query {i}"]
            )
            for i in range(5)
        ]

        # Execute requests concurrently
        tasks = [
            integration_generation_service.generate_answer(req) for req in requests
        ]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0

    def test_service_health_check_after_errors(
        self, integration_generation_service, mocker
    ):
        """Test health check after service encounters errors."""
        # Service should still be healthy after mock errors
        assert integration_generation_service.is_healthy() is True

        # Simulate component failure
        mocker.patch.object(integration_generation_service, "rag_chain", None)
        assert integration_generation_service.is_healthy() is False

    def test_service_resource_management(
        self, integration_settings, mock_langchain_components
    ):
        """Test service resource management and cleanup."""
        # Create multiple service instances
        services = [GenerationService(settings=integration_settings) for _ in range(3)]

        # All should initialize successfully
        for service in services:
            assert service.is_healthy() is True
            assert service.settings == integration_settings

    @pytest.mark.asyncio
    async def test_generate_answer_with_various_query_lengths(
        self, integration_generation_service
    ):
        """Test generate_answer with various query lengths."""
        query_lengths = [1, 10, 100, 1000]

        for length in query_lengths:
            query = "a" * length
            request = GenerateRequest(query=query, context_chunks=["Some context"])

            result = await integration_generation_service.generate_answer(request)

            assert isinstance(result, str)
            assert len(result) > 0


class TestGenerationServiceResourceUsage:
    """Integration tests for resource usage and performance characteristics."""

    def test_service_memory_usage(
        self, integration_generation_service, sample_requests_for_service
    ):
        """Test service memory usage patterns."""
        # This is a placeholder for memory usage testing
        # In a real scenario, you might use memory profiling tools
        assert integration_generation_service.is_healthy() is True

    @pytest.mark.asyncio
    async def test_service_processing_time_patterns(
        self, integration_generation_service
    ):
        """Test service processing time for different input sizes."""
        import time

        small_request = GenerateRequest(
            query="Short query", context_chunks=["Short context"]
        )

        large_request = GenerateRequest(
            query="Long query " * 100,
            context_chunks=["Long context " * 100 for _ in range(10)],
        )

        # Test small request
        start_time = time.time()
        result1 = await integration_generation_service.generate_answer(small_request)
        small_time = time.time() - start_time

        # Test large request
        start_time = time.time()
        result2 = await integration_generation_service.generate_answer(large_request)
        large_time = time.time() - start_time

        # Both should complete successfully
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert len(result1) > 0
        assert len(result2) > 0

        # Times should be reasonable (not hanging)
        assert small_time < 60  # Should complete within 60 seconds
        assert large_time < 60  # Should complete within 60 seconds
