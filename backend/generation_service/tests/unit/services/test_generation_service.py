"""
Unit tests for the GenerationService.
"""

from unittest.mock import Mock

import pytest
from app.models import GenerateRequest
from app.services.generation import GenerationService
from fastapi import HTTPException, status


class TestGenerationServiceInitialization:
    """Test cases for GenerationService initialization."""

    def test_service_initialization_success(self, unit_settings, mocker):
        """Test successful service initialization."""
        mock_chat_model = Mock()
        mocker.patch("app.services.generation.ChatOpenAI", return_value=mock_chat_model)

        service = GenerationService(settings=unit_settings)

        assert service.settings == unit_settings
        assert service.chat_model == mock_chat_model
        assert service.rag_chain is not None

    def test_service_initialization_invalid_provider(self, unit_settings, mocker):
        """Test service initialization with invalid LLM provider."""
        unit_settings.LLM_PROVIDER = "invalid_provider"

        with pytest.raises(RuntimeError) as exc_info:
            GenerationService(settings=unit_settings)

        assert "Unsupported LLM provider" in str(exc_info.value)

    def test_service_initialization_missing_api_key(self, unit_settings, mocker):
        """Test service initialization with missing API key."""
        unit_settings.OPENAI_API_KEY = None

        with pytest.raises(RuntimeError) as exc_info:
            GenerationService(settings=unit_settings)

        assert "OPENAI_API_KEY is required" in str(exc_info.value)


class TestHealthCheck:
    """Test cases for is_healthy method."""

    def test_is_healthy_all_components_present(self, mocked_generation_service):
        """Test is_healthy returns True when all components are initialized."""
        # Ensure all components are properly set
        mocked_generation_service.rag_chain = Mock()
        mocked_generation_service.chat_model = Mock()
        mocked_generation_service.prompt_template = Mock()
        mocked_generation_service.output_parser = Mock()

        assert mocked_generation_service.is_healthy() is True

    def test_is_healthy_missing_rag_chain(self, mocked_generation_service):
        """Test is_healthy returns False when rag_chain is missing."""
        mocked_generation_service.rag_chain = None
        mocked_generation_service.chat_model = Mock()
        mocked_generation_service.prompt_template = Mock()
        mocked_generation_service.output_parser = Mock()

        assert mocked_generation_service.is_healthy() is False

    def test_is_healthy_missing_chat_model(self, mocked_generation_service):
        """Test is_healthy returns False when chat_model is missing."""
        mocked_generation_service.rag_chain = Mock()
        mocked_generation_service.chat_model = None
        mocked_generation_service.prompt_template = Mock()
        mocked_generation_service.output_parser = Mock()

        assert mocked_generation_service.is_healthy() is False

    def test_is_healthy_missing_prompt_template(self, mocked_generation_service):
        """Test is_healthy returns False when prompt_template is missing."""
        mocked_generation_service.rag_chain = Mock()
        mocked_generation_service.chat_model = Mock()
        mocked_generation_service.prompt_template = None
        mocked_generation_service.output_parser = Mock()

        assert mocked_generation_service.is_healthy() is False

    def test_is_healthy_missing_output_parser(self, mocked_generation_service):
        """Test is_healthy returns False when output_parser is missing."""
        mocked_generation_service.rag_chain = Mock()
        mocked_generation_service.chat_model = Mock()
        mocked_generation_service.prompt_template = Mock()
        mocked_generation_service.output_parser = None

        assert mocked_generation_service.is_healthy() is False

    def test_is_healthy_multiple_missing_components(self, mocked_generation_service):
        """Test is_healthy returns False when multiple components are missing."""
        mocked_generation_service.rag_chain = None
        mocked_generation_service.chat_model = None
        mocked_generation_service.prompt_template = Mock()
        mocked_generation_service.output_parser = Mock()

        assert mocked_generation_service.is_healthy() is False

    def test_is_healthy_exception_handling(self, unit_settings, mocker):
        """Test is_healthy handles exceptions gracefully when components fail."""
        # Create a service where one component initialization fails
        mock_chat_model = Mock()
        mocker.patch("app.services.generation.ChatOpenAI", return_value=mock_chat_model)

        service = GenerationService(settings=unit_settings)

        # Create a scenario where accessing a component might raise an exception
        # by mocking the all() function to raise an exception

        def failing_all(iterable):
            raise Exception("Unexpected error during health check")

        mocker.patch("builtins.all", side_effect=failing_all)

        result = service.is_healthy()
        assert result is False


class TestPromptTemplateUsage:
    """Test cases for prompt template integration."""

    @pytest.mark.asyncio
    async def test_prompt_template_receives_correct_variables(
        self, generation_service_with_mock_chain, sample_generate_request
    ):
        """Test that prompt template receives the expected context and query variables."""
        service, mock_chain = generation_service_with_mock_chain
        mock_chain.ainvoke.return_value = "Test response"

        await service.generate_answer(sample_generate_request)

        # Verify the chain was called with the expected structure
        call_args = mock_chain.ainvoke.call_args[0][0]
        assert "context" in call_args
        assert "query" in call_args
        assert call_args["query"] == sample_generate_request.query

        # Verify context formatting
        expected_context = "\n---\n".join(sample_generate_request.context_chunks)
        assert call_args["context"] == expected_context

    @pytest.mark.asyncio
    async def test_context_formatting_in_chain_call(
        self, generation_service_with_mock_chain
    ):
        """Test context formatting is applied correctly before chain invocation."""
        service, mock_chain = generation_service_with_mock_chain
        mock_chain.ainvoke.return_value = "Formatted response"

        request = GenerateRequest(
            query="Test query", context_chunks=["Chunk 1", "Chunk 2", "Chunk 3"]
        )

        await service.generate_answer(request)

        call_args = mock_chain.ainvoke.call_args[0][0]
        expected_context = "Chunk 1\n---\nChunk 2\n---\nChunk 3"
        assert call_args["context"] == expected_context


class TestLLMInitialization:
    """Test cases for LLM initialization edge cases."""

    def test_openai_model_configuration(self, unit_settings, mocker):
        """Test OpenAI model is configured with correct parameters."""
        mock_chat_openai = mocker.patch("app.services.generation.ChatOpenAI")

        service = GenerationService(settings=unit_settings)

        # Verify ChatOpenAI was called with expected parameters
        mock_chat_openai.assert_called_once_with(
            model=unit_settings.LLM_MODEL_NAME,
            temperature=unit_settings.LLM_TEMPERATURE,
            max_tokens=unit_settings.LLM_MAX_TOKENS,
            openai_api_key=unit_settings.OPENAI_API_KEY.get_secret_value(),
        )

    def test_openai_initialization_failure(self, unit_settings, mocker):
        """Test handling of OpenAI initialization failure."""
        mocker.patch(
            "app.services.generation.ChatOpenAI",
            side_effect=Exception("OpenAI API connection failed"),
        )

        with pytest.raises(RuntimeError) as exc_info:
            GenerationService(settings=unit_settings)

        assert "Failed to initialize OpenAI model" in str(exc_info.value)


class TestFormatContext:
    """Test cases for _format_context method."""

    @pytest.mark.parametrize(
        "chunks, expected_output",
        [
            ([], "No context provided."),
            (["First chunk."], "First chunk."),
            (["Chunk 1.", "Chunk 2."], "Chunk 1.\n---\nChunk 2."),
            (["Line 1\nLine 2", "Chunk 2"], "Line 1\nLine 2\n---\nChunk 2"),
            (["", "Valid chunk"], "\n---\nValid chunk"),
        ],
        ids=["empty", "single", "multiple", "multiline", "empty_chunk"],
    )
    def test_format_context(self, mocked_generation_service, chunks, expected_output):
        """Tests the _format_context method with various inputs."""
        formatted = mocked_generation_service._format_context(chunks)
        assert formatted == expected_output

    def test_format_context_large_number_of_chunks(self, mocked_generation_service):
        """Test _format_context with many chunks."""
        chunks = [f"Chunk {i}" for i in range(100)]
        formatted = mocked_generation_service._format_context(chunks)

        assert "Chunk 0" in formatted
        assert "Chunk 99" in formatted
        assert formatted.count("---") == 99  # 100 chunks means 99 separators


class TestGenerateAnswer:
    """Test cases for generate_answer method."""

    @pytest.mark.asyncio
    async def test_generate_answer_success(
        self, generation_service_with_mock_chain, sample_generate_request
    ):
        """Test generate_answer successful path."""
        service, mock_chain = generation_service_with_mock_chain
        expected_answer = "FastAPI is a modern Python web framework."
        mock_chain.ainvoke.return_value = expected_answer

        result = await service.generate_answer(sample_generate_request)

        assert result == expected_answer
        mock_chain.ainvoke.assert_awaited_once()

        # Verify the chain was called with correct input structure
        call_args = mock_chain.ainvoke.call_args[0][0]
        assert "context" in call_args
        assert "query" in call_args
        assert call_args["query"] == sample_generate_request.query

    @pytest.mark.asyncio
    async def test_generate_answer_with_empty_context(
        self, generation_service_with_mock_chain
    ):
        """Test generate_answer with empty context chunks."""
        service, mock_chain = generation_service_with_mock_chain
        expected_answer = "I don't have enough information to answer that."
        mock_chain.ainvoke.return_value = expected_answer

        request = GenerateRequest(query="What is FastAPI?", context_chunks=[])
        result = await service.generate_answer(request)

        assert result == expected_answer
        call_args = mock_chain.ainvoke.call_args[0][0]
        assert call_args["context"] == "No context provided."

    @pytest.mark.asyncio
    async def test_generate_answer_llm_timeout_error(
        self, generation_service_with_mock_chain, sample_generate_request
    ):
        """Test generate_answer when LLM times out."""
        service, mock_chain = generation_service_with_mock_chain
        mock_chain.ainvoke.side_effect = Exception("timeout")

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(sample_generate_request)

        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "timed out" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_generate_answer_rate_limit_error(
        self, generation_service_with_mock_chain, sample_generate_request
    ):
        """Test generate_answer when rate limited."""
        service, mock_chain = generation_service_with_mock_chain
        mock_chain.ainvoke.side_effect = Exception("rate limit exceeded")

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(sample_generate_request)

        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "rate limit" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_generate_answer_authentication_error(
        self, generation_service_with_mock_chain, sample_generate_request
    ):
        """Test generate_answer when authentication fails."""
        service, mock_chain = generation_service_with_mock_chain
        mock_chain.ainvoke.side_effect = Exception("authentication failed")

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(sample_generate_request)

        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "authentication" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_generate_answer_generic_error(
        self, generation_service_with_mock_chain, sample_generate_request
    ):
        """Test generate_answer with generic error."""
        service, mock_chain = generation_service_with_mock_chain
        error_message = "Unknown error occurred"
        mock_chain.ainvoke.side_effect = Exception(error_message)

        with pytest.raises(HTTPException) as exc_info:
            await service.generate_answer(sample_generate_request)

        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert error_message in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_generate_answer_with_complex_query(
        self, generation_service_with_mock_chain
    ):
        """Test generate_answer with complex query and multiple context chunks."""
        service, mock_chain = generation_service_with_mock_chain
        expected_answer = (
            "Based on the provided context, here's a comprehensive answer..."
        )
        mock_chain.ainvoke.return_value = expected_answer

        complex_request = GenerateRequest(
            query="Can you explain the differences between FastAPI and Flask, including performance characteristics?",
            context_chunks=[
                "FastAPI is a modern web framework with automatic API documentation.",
                "Flask is a lightweight WSGI web application framework.",
                "FastAPI supports async/await natively and is faster than Flask.",
                "Both frameworks are popular choices for Python web development.",
            ],
        )

        result = await service.generate_answer(complex_request)

        assert result == expected_answer
        call_args = mock_chain.ainvoke.call_args[0][0]
        assert len(call_args["context"]) > 100  # Should be a long context
        assert "---" in call_args["context"]  # Should have separators


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    def test_initialize_llm_openai_success(self, unit_settings, mocker):
        """Test successful OpenAI LLM initialization."""
        mock_chat_openai = Mock()
        mocker.patch(
            "app.services.generation.ChatOpenAI", return_value=mock_chat_openai
        )

        service = GenerationService(settings=unit_settings)
        result = service._initialize_llm()

        assert result == mock_chat_openai

    def test_initialize_llm_invalid_provider(self, unit_settings):
        """Test LLM initialization with invalid provider."""
        unit_settings.LLM_PROVIDER = "invalid"
        service = GenerationService.__new__(
            GenerationService
        )  # Create instance without __init__
        service.settings = unit_settings

        with pytest.raises(ValueError) as exc_info:
            service._initialize_llm()

        assert "Unsupported LLM provider: invalid" in str(exc_info.value)

    def test_initialize_llm_missing_api_key(self, unit_settings):
        """Test LLM initialization with missing API key."""
        unit_settings.OPENAI_API_KEY = None
        service = GenerationService.__new__(GenerationService)
        service.settings = unit_settings

        with pytest.raises(ValueError) as exc_info:
            service._initialize_llm()

        assert "OPENAI_API_KEY is required" in str(exc_info.value)
