from pathlib import Path

import pytest
from app.config import Settings
from app.deps import (
    get_collection_manager_service,
    get_file_management_service,
    get_ingestion_processor_service,
    get_ingestion_state_service,
    get_settings,
)
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def integration_settings(test_data_root, base_settings, tmp_path):
    """Settings for integration tests with real components."""
    chroma_local_path = tmp_path / "chroma_integration"
    chroma_local_path.mkdir(exist_ok=True)

    # Create a source directory with some test data
    source_dir = test_data_root / "integration"
    source_dir.mkdir(exist_ok=True)

    return Settings(
        **base_settings,
        SOURCE_DIRECTORY=str(source_dir),
        CHROMA_MODE="local",
        CHROMA_LOCAL_PATH=str(chroma_local_path),
        CHROMA_COLLECTION_NAME="test_integration_collection",
        CLEAN_COLLECTION_BEFORE_INGEST=True,  # Clean between integration tests
    )


@pytest.fixture
def test_client(integration_settings, mocker):
    """Basic test client with integration settings and mocked startup."""
    # Mock the manager classes to prevent real connections during startup
    mock_chroma_manager = mocker.Mock()
    mock_embedding_manager = mocker.Mock()
    mock_vector_store_manager = mocker.Mock()
    mock_state_service = mocker.Mock()

    # Mock the manager constructors
    mocker.patch("app.main.ChromaClientManager", return_value=mock_chroma_manager)
    mocker.patch("app.main.EmbeddingModelManager", return_value=mock_embedding_manager)
    mocker.patch("app.main.VectorStoreManager", return_value=mock_vector_store_manager)
    mocker.patch("app.main.IngestionStateService", return_value=mock_state_service)

    app.dependency_overrides[get_settings] = lambda: integration_settings

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def mock_state_service(mocker):
    """Mock ingestion state service with common default behavior."""
    mock_service = mocker.Mock()
    mock_service.is_processing.return_value = False
    mock_service.start_ingestion.return_value = True
    mock_service.get_status.return_value = {
        "is_processing": False,
        "last_completed": None,
        "current_status": "idle",
        "documents_processed": 0,
        "chunks_added": 0,
        "errors": [],
    }
    return mock_service


@pytest.fixture
def mock_file_service(mocker):
    """Mock file management service with common default behavior."""
    mock_service = mocker.Mock()
    mock_service.list_documents.return_value = [
        {"name": "document1.pdf", "size": 1024},
        {"name": "document2.pdf", "size": 2048},
    ]
    mock_service.count_documents.return_value = 2
    mock_service.save_uploaded_file.return_value = {
        "name": "uploaded.pdf",
        "size": 1500,
    }
    mock_service.has_duplicate_filename.return_value = False
    return mock_service


@pytest.fixture
def mock_collection_service(mocker):
    """Mock collection management service with common default behavior."""
    mock_service = mocker.Mock()
    mock_service.clear_collection_and_documents.return_value = {
        "collection_cleared": True,
        "documents_cleared": True,
        "messages": [
            "Collection cleared successfully",
            "Documents cleared successfully",
        ],
    }
    return mock_service


@pytest.fixture
def mock_ingestion_processor(mocker):
    """Mock ingestion processor with common default behavior."""
    mock_processor = mocker.Mock()
    mock_status = mocker.Mock()
    mock_status.documents_processed = 2
    mock_status.chunks_added = 10
    mock_status.errors = []
    mock_processor.run_ingestion.return_value = mock_status
    return mock_processor


@pytest.fixture
def client_with_mocked_services(
    integration_settings,
    mock_state_service,
    mock_file_service,
    mock_collection_service,
    mock_ingestion_processor,
    mocker,
):
    """Test client with all services mocked at both startup and dependency levels."""
    # Mock the manager classes to prevent real connections during startup
    mock_chroma_manager = mocker.Mock()
    mock_embedding_manager = mocker.Mock()
    mock_vector_store_manager = mocker.Mock()

    # Mock the manager constructors to prevent startup issues
    mocker.patch("app.main.ChromaClientManager", return_value=mock_chroma_manager)
    mocker.patch("app.main.EmbeddingModelManager", return_value=mock_embedding_manager)
    mocker.patch("app.main.VectorStoreManager", return_value=mock_vector_store_manager)
    mocker.patch("app.main.IngestionStateService", return_value=mock_state_service)

    # Override dependency injection
    app.dependency_overrides[get_settings] = lambda: integration_settings
    app.dependency_overrides[get_ingestion_state_service] = (
        lambda request: mock_state_service
    )
    app.dependency_overrides[get_file_management_service] = (
        lambda **kwargs: mock_file_service
    )
    app.dependency_overrides[get_collection_manager_service] = (
        lambda **kwargs: mock_collection_service
    )
    app.dependency_overrides[get_ingestion_processor_service] = (
        lambda **kwargs: mock_ingestion_processor
    )

    with TestClient(app) as client:
        yield (
            client,
            {
                "state_service": mock_state_service,
                "file_service": mock_file_service,
                "collection_service": mock_collection_service,
                "processor": mock_ingestion_processor,
            },
        )

    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for integration tests."""
    return "This is a sample PDF document for integration testing. It contains multiple sentences to test document splitting and embedding functionality."


@pytest.fixture
def test_pdf_file(integration_settings, sample_pdf_content, tmp_path):
    """Create a test PDF file for integration tests."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = Path(integration_settings.SOURCE_DIRECTORY) / "test_document.pdf"

    # Create a simple PDF with the sample content
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, sample_pdf_content)
    c.save()

    yield pdf_path

    # Cleanup
    if pdf_path.exists():
        pdf_path.unlink()
