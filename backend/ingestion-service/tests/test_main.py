from pathlib import Path  # Import Path
from unittest.mock import MagicMock, patch

import pytest
from app.config import Settings  # To create mock settings objects
from app.main import lifespan  # The lifespan function from your ingestion service

# Mark all tests in this module as async, as lifespan is an async context manager
pytestmark = pytest.mark.asyncio


# Helper to create a base for mock settings dictionary, ensuring all necessary fields are present
def get_base_mock_settings_dict(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """
    Provides a base dictionary for creating Settings objects.
    It includes all fields typically set by override_settings in conftest.py,
    allowing tests to customize specific fields like CHROMA_MODE, CHROMA_LOCAL_PATH, CHROMA_HOST.
    """
    source_dir = tmp_path_factory.mktemp("test_source_docs_main")
    default_chroma_local_path = tmp_path_factory.mktemp("chroma_default_main")

    # These values should mirror the kind of defaults you'd want for most unit tests,
    # or be consistent with conftest.py's override_settings where appropriate.
    return {
        "SOURCE_DIRECTORY": source_dir,
        "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",  # Consistent with conftest
        "CHUNK_SIZE": 100,  # Consistent with conftest
        "CHUNK_OVERLAP": 20,  # Consistent with conftest
        "CHROMA_COLLECTION_NAME": f"test_main_collection_{Path(source_dir).name}",  # Unique per test run potentially
        "CLEAN_COLLECTION_BEFORE_INGEST": False,
        "CHROMA_MODE": "local",  # Default mode
        "CHROMA_LOCAL_PATH": str(default_chroma_local_path),  # Default for local mode
        "CHROMA_HOST": None,  # Default for docker mode (i.e., not set if local)
        # Add any other fields required by your Settings model that don't have defaults
        # For example, if LOG_LEVEL or APP_ENV are in Settings and lack defaults:
        # "LOG_LEVEL": "INFO",
        # "APP_ENV": "test",
    }


async def test_lifespan_ingestion_local_missing_path(
    mocker, tmp_path_factory: pytest.TempPathFactory
):
    """Test lifespan raises RuntimeError if local mode is chosen and CHROMA_LOCAL_PATH is None."""
    mock_settings_dict = get_base_mock_settings_dict(tmp_path_factory)
    mock_settings_dict.update(
        {
            "CHROMA_MODE": "local",
            "CHROMA_LOCAL_PATH": None,  # Crucial for this test
            "CHROMA_HOST": "http://dummyhost:8000",  # Should be ignored in local mode
        }
    )
    mocker.patch("app.main.settings", Settings(**mock_settings_dict))

    with (
        patch("app.services.ingestion_processor._embedding_model", None),
        patch("app.services.ingestion_processor._chroma_client", None),
        patch("app.services.ingestion_processor._vector_store", None),
    ):
        # The error message in your app.main.lifespan might refer to 'chroma_path' generically
        # or specifically 'chroma_local_path'. Adjust match if needed.
        with pytest.raises(
            RuntimeError,
            match="Failed to pre-load resources: chroma_local_path is required for local mode.",
        ):
            async with lifespan(app=MagicMock()):
                pass


async def test_lifespan_ingestion_docker_missing_host(
    mocker, tmp_path_factory: pytest.TempPathFactory
):
    """Test lifespan raises RuntimeError if docker mode is chosen and CHROMA_HOST is None."""
    mock_settings_dict = get_base_mock_settings_dict(tmp_path_factory)
    mock_settings_dict.update(
        {
            "CHROMA_MODE": "docker",
            "CHROMA_HOST": None,  # Crucial for this test
            "CHROMA_LOCAL_PATH": str(
                tmp_path_factory.mktemp("dummy_local_path")
            ),  # Should be ignored
        }
    )
    mocker.patch("app.main.settings", Settings(**mock_settings_dict))

    with (
        patch("app.services.ingestion_processor._embedding_model", None),
        patch("app.services.ingestion_processor._chroma_client", None),
        patch("app.services.ingestion_processor._vector_store", None),
    ):
        with pytest.raises(
            RuntimeError,
            match="Failed to pre-load resources: chroma_host is required for docker mode.",
        ):
            async with lifespan(app=MagicMock()):
                pass


async def test_lifespan_ingestion_local_success(
    mocker, tmp_path_factory: pytest.TempPathFactory
):
    """Test successful resource initialization in local mode (dependencies mocked)."""
    # get_base_mock_settings_dict already sets up for local mode by default
    mock_settings_dict = get_base_mock_settings_dict(tmp_path_factory)
    # Ensure CHROMA_LOCAL_PATH is explicitly valid if not already by default base
    mock_settings_dict["CHROMA_LOCAL_PATH"] = str(
        tmp_path_factory.mktemp("local_success_path")
    )
    mock_settings_dict["CHROMA_MODE"] = "local"
    mock_settings_dict["CHROMA_HOST"] = None

    mock_settings_obj = Settings(**mock_settings_dict)
    mocker.patch("app.main.settings", mock_settings_obj)

    mock_embedding_instance = MagicMock()
    mock_chroma_instance = MagicMock()

    with (
        patch(
            "app.services.ingestion_processor.get_embedding_model",
            return_value=mock_embedding_instance,
        ) as mock_get_model,
        patch(
            "app.services.ingestion_processor.get_chroma_client",
            return_value=mock_chroma_instance,
        ) as mock_get_chroma,
    ):
        async with lifespan(app=MagicMock()):
            pass

        mock_get_model.assert_called_once_with(mock_settings_obj)
        mock_get_chroma.assert_called_once_with(mock_settings_obj)


async def test_lifespan_ingestion_docker_success(
    mocker, tmp_path_factory: pytest.TempPathFactory
):
    """Test successful resource initialization in docker mode (dependencies mocked)."""
    mock_settings_dict = get_base_mock_settings_dict(tmp_path_factory)
    mock_settings_dict.update(
        {
            "CHROMA_MODE": "docker",
            "CHROMA_HOST": "http://mockdockerhost:8010",  # Use a port consistent with typical Chroma setup
            "CHROMA_LOCAL_PATH": None,  # Explicitly None for docker mode
        }
    )
    mock_settings_obj = Settings(**mock_settings_dict)
    mocker.patch("app.main.settings", mock_settings_obj)

    mock_embedding_instance = MagicMock()
    mock_chroma_instance = MagicMock()

    with (
        patch(
            "app.services.ingestion_processor.get_embedding_model",
            return_value=mock_embedding_instance,
        ) as mock_get_model,
        patch(
            "app.services.ingestion_processor.get_chroma_client",
            return_value=mock_chroma_instance,
        ) as mock_get_chroma,
    ):
        async with lifespan(app=MagicMock()):
            pass

        mock_get_model.assert_called_once_with(mock_settings_obj)
        mock_get_chroma.assert_called_once_with(mock_settings_obj)


async def test_lifespan_ingestion_embedding_model_load_failure(
    mocker, tmp_path_factory: pytest.TempPathFactory
):
    """Test lifespan handles failure during embedding model loading."""
    mock_settings_dict = get_base_mock_settings_dict(tmp_path_factory)
    # Ensure it's a valid local setup up to the point of model loading
    mock_settings_dict["CHROMA_MODE"] = "local"
    mock_settings_dict["CHROMA_LOCAL_PATH"] = str(
        tmp_path_factory.mktemp("model_fail_local_path")
    )
    mock_settings_dict["CHROMA_HOST"] = None

    mock_settings_obj = Settings(**mock_settings_dict)
    mocker.patch("app.main.settings", mock_settings_obj)

    with (
    )
      with (
          patch(
              "app.services.ingestion_processor.get_embedding_model",
              side_effect=RuntimeError("Simulated model load error"),
          ) as mock_get_model,
          patch("app.services.ingestion_processor.get_chroma_client") as mock_get_chroma,
      ):
          with pytest.raises(
              RuntimeError,
              match="Failed to pre-load resources: Simulated model load error",
          ):
              async with lifespan(app=MagicMock()):
                  pass
          mock_get_model.assert_called_once_with(mock_settings_obj)
          mock_get_chroma.assert_not_called()  # Should not be called if model loading fails first
