import pytest
from app.config import Settings
# from app.services.ingestion_processor import IngestionProcessorService  # Commented out to avoid hanging


@pytest.fixture
def unit_settings(test_data_root, base_settings, tmp_path):
    """Lightweight settings for unit tests with valid CHROMA_MODE."""
    chroma_local_path = tmp_path / "chroma_unit"
    chroma_local_path.mkdir(exist_ok=True)

    return Settings(
        **base_settings,
        SOURCE_DIRECTORY=str(test_data_root / "unit"),
        CHROMA_MODE="local",  # Use local mode for unit tests
        CHROMA_PATH=str(chroma_local_path),
        CHROMA_COLLECTION_NAME="test_unit_collection",
    )


@pytest.fixture
def mock_chroma_client(mocker):
    """Mock ChromaDB client."""
    mock_client = mocker.Mock()
    mock_client.delete_collection = mocker.Mock()
    return mock_client


@pytest.fixture
def mock_chroma_vector_store(mocker):
    """Mock Chroma vector store."""
    mock_store = mocker.Mock()
    mock_store.add_documents = mocker.Mock()
    mock_store._collection = mocker.Mock()
    mock_store._collection.get = mocker.Mock(return_value={"metadatas": []})
    return mock_store


# @pytest.fixture
# def mocked_ingestion_service(
#     unit_settings: Settings,
#     mock_chroma_vector_store,
#     mock_chroma_client,
#     mocker,
# ):
#     """Provides an IngestionProcessorService with mocked internal components."""
#     if IngestionProcessorService is None:
#         pytest.skip("IngestionProcessorService not available")
#
#     # Mock the manager classes
#     mock_chroma_manager = mocker.Mock()
#     mock_chroma_manager.get_client.return_value = mock_chroma_client

#     mock_embedding_manager = mocker.Mock()

#     mock_vector_store_manager = mocker.Mock()
#     mock_vector_store_manager.get_vector_store.return_value = mock_chroma_vector_store
#     mock_vector_store_manager.reset = mocker.Mock()

#     # Instantiate the service with mocked managers
#     service = IngestionProcessorService(
#         settings=unit_settings,
#         chroma_manager=mock_chroma_manager,
#         embedding_manager=mock_embedding_manager,
#         vector_store_manager=mock_vector_store_manager,
#     )
#     return service
