import pytest
from app.services.ingestion_processor import IngestionProcessorService
from app.services.ingestion_state import IngestionStateService


@pytest.fixture
def mock_service_dependencies(mocker):
    """Common mocked dependencies for services."""
    return {
        "chroma_manager": mocker.Mock(),
        "embedding_manager": mocker.Mock(),
        "vector_store_manager": mocker.Mock(),
        "file_management": mocker.Mock(),
    }


@pytest.fixture
def ingestion_processor_service(unit_settings, mock_service_dependencies):
    """Pre-configured IngestionProcessorService for testing."""
    return IngestionProcessorService(
        settings=unit_settings,
        chroma_manager=mock_service_dependencies["chroma_manager"],
        embedding_manager=mock_service_dependencies["embedding_manager"],
        vector_store_manager=mock_service_dependencies["vector_store_manager"],
    )


@pytest.fixture
def ingestion_state_service():
    """Fresh IngestionStateService for each test."""
    return IngestionStateService()
