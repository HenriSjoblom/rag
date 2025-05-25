"""
Unit tests for EmbeddingModelManager.
"""

from unittest.mock import Mock

import pytest
from app.config import Settings
from app.services.embedding_manager import EmbeddingModelManager


class TestEmbeddingModelManagerInit:
    """Test initialization of EmbeddingModelManager."""

    def test_init_with_settings(self, mock_settings):
        """Test initialization with settings."""
        manager = EmbeddingModelManager(mock_settings)
        assert manager.settings == mock_settings
        assert manager._model is None

    def test_init_stores_settings_reference(self, mock_settings):
        """Test that settings reference is stored correctly."""
        manager = EmbeddingModelManager(mock_settings)
        assert manager.settings is mock_settings


class TestEmbeddingModelManagerGetModel:
    """Test get_model functionality."""

    def test_get_model_success(self, mock_settings, mocker):
        """Test successful model loading."""
        mock_model_instance = Mock()
        mock_sentence_transformer = mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)
        model = manager.get_model()

        assert model == mock_model_instance
        assert manager._model == mock_model_instance
        mock_sentence_transformer.assert_called_once_with("test-model")

    def test_get_model_cached(self, mock_settings, mocker):
        """Test that model is cached after first call."""
        mock_model_instance = Mock()
        mock_sentence_transformer = mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)

        # First call
        model1 = manager.get_model()
        # Second call
        model2 = manager.get_model()

        assert model1 == model2
        assert model1 == mock_model_instance
        # Should only be called once due to caching
        mock_sentence_transformer.assert_called_once()

    def test_get_model_file_not_found_error(self, mock_settings, mocker):
        """Test handling FileNotFoundError when loading model."""
        mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            side_effect=FileNotFoundError("Model file not found"),
        )

        manager = EmbeddingModelManager(mock_settings)

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_model()

        assert "not found" in str(exc_info.value)
        assert "test-model" in str(exc_info.value)
        assert "Please check the model name" in str(exc_info.value)

    def test_get_model_memory_error(self, mock_settings, mocker):
        """Test handling MemoryError when loading model."""
        mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            side_effect=MemoryError("Out of memory"),
        )

        manager = EmbeddingModelManager(mock_settings)

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_model()

        assert "Insufficient memory" in str(exc_info.value)
        assert "test-model" in str(exc_info.value)

    def test_get_model_general_exception(self, mock_settings, mocker):
        """Test handling general exceptions when loading model."""
        mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            side_effect=Exception("Model loading failed"),
        )

        manager = EmbeddingModelManager(mock_settings)

        with pytest.raises(RuntimeError) as exc_info:
            manager.get_model()

        assert "Failed to load embedding model" in str(exc_info.value)
        assert "test-model" in str(exc_info.value)
        assert "Model loading failed" in str(exc_info.value)

    def test_get_model_different_model_names(self, mocker):
        """Test with different model names."""
        test_cases = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ]

        for model_name in test_cases:
            settings = Settings(
                EMBEDDING_MODEL_NAME=model_name,
                CHROMA_COLLECTION_NAME="test_collection",
                TOP_K_RESULTS=3,
                CHROMA_MODE="local",
                CHROMA_PATH="/tmp/test",
            )

            mock_model_instance = Mock()
            mock_sentence_transformer = mocker.patch(
                "app.services.embedding_manager.SentenceTransformer",
                return_value=mock_model_instance,
            )

            manager = EmbeddingModelManager(settings)
            model = manager.get_model()

            assert model == mock_model_instance
            mock_sentence_transformer.assert_called_with(model_name)
            mock_sentence_transformer.reset_mock()


class TestEmbeddingModelManagerReset:
    """Test reset functionality."""

    def test_reset_with_loaded_model(self, mock_settings, mocker):
        """Test reset with a loaded model."""
        mock_model_instance = Mock()
        mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)

        # Load model first
        manager.get_model()
        assert manager._model == mock_model_instance

        # Reset
        manager.reset()
        assert manager._model is None

    def test_reset_with_no_model(self, mock_settings):
        """Test reset when no model is loaded."""
        manager = EmbeddingModelManager(mock_settings)
        assert manager._model is None

        # Reset should not raise exception
        manager.reset()
        assert manager._model is None

    def test_reset_exception_handling(self, mock_settings, mocker):
        """Test reset when an exception occurs during reset."""
        # Mock the logger to avoid import issues
        mock_logger = mocker.patch("app.services.embedding_manager.logger")

        manager = EmbeddingModelManager(mock_settings)

        # Simulate an exception during reset by mocking setattr to fail
        def failing_reset():
            try:
                # Simulate some operation that fails during reset
                raise Exception("Reset error")
            except Exception as e:
                mock_logger.error(
                    f"Error during embedding model manager reset: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Failed to reset embedding model manager: {str(e)}"
                ) from e

        manager.reset = failing_reset

        with pytest.raises(RuntimeError) as exc_info:
            manager.reset()

        assert "Failed to reset embedding model manager" in str(exc_info.value)
        assert "Reset error" in str(exc_info.value)

    def test_reset_after_model_loaded_and_reset_again(self, mock_settings, mocker):
        """Test multiple reset operations."""
        mock_model_instance = Mock()
        mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)

        # Load model
        manager.get_model()
        assert manager._model == mock_model_instance

        # First reset
        manager.reset()
        assert manager._model is None

        # Second reset
        manager.reset()
        assert manager._model is None


class TestEmbeddingModelManagerEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_get_model_after_reset_loads_fresh_model(self, mock_settings, mocker):
        """Test that get_model after reset loads a fresh model instance."""
        mock_model_instance1 = Mock()
        mock_model_instance2 = Mock()
        mock_sentence_transformer = mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            side_effect=[mock_model_instance1, mock_model_instance2],
        )

        manager = EmbeddingModelManager(mock_settings)

        # First load
        model1 = manager.get_model()
        assert model1 == mock_model_instance1

        # Reset
        manager.reset()
        assert manager._model is None

        # Second load
        model2 = manager.get_model()
        assert model2 == mock_model_instance2
        assert model1 != model2  # Should be different instances

        # Should be called twice
        assert mock_sentence_transformer.call_count == 2

    def test_multiple_get_model_calls_after_error(self, mock_settings, mocker):
        """Test behavior when get_model is called multiple times after an error."""
        # First call fails
        mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            side_effect=[Exception("Loading failed"), Mock()],
        )

        manager = EmbeddingModelManager(mock_settings)

        # First call should fail
        with pytest.raises(RuntimeError):
            manager.get_model()

        assert manager._model is None

        # Second call should succeed
        model = manager.get_model()
        assert model is not None
        assert manager._model == model

    def test_concurrent_get_model_simulation(self, mock_settings, mocker):
        """Test simulation of concurrent get_model calls."""
        mock_model_instance = Mock()
        mock_sentence_transformer = mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)

        # Simulate multiple concurrent calls
        models = []
        for _ in range(5):
            models.append(manager.get_model())

        # All should return the same cached instance
        assert all(model == mock_model_instance for model in models)
        # SentenceTransformer should only be called once
        mock_sentence_transformer.assert_called_once()

    def test_settings_changes_dont_affect_loaded_model(self, mock_settings, mocker):
        """Test that changing settings doesn't affect already loaded model."""
        mock_model_instance = Mock()
        mocker.patch(
            "app.services.embedding_manager.SentenceTransformer",
            return_value=mock_model_instance,
        )

        manager = EmbeddingModelManager(mock_settings)

        # Load model
        model = manager.get_model()
        assert model == mock_model_instance

        # Change settings (simulating settings modification)
        manager.settings.EMBEDDING_MODEL_NAME = "different-model"

        # Get model again - should still return cached model
        model2 = manager.get_model()
        assert model2 == mock_model_instance
        assert model2 == model  # Same instance
