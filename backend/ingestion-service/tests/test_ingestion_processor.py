import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import chromadb

from app.services.ingestion_processor import IngestionProcessorService, IngestionStatus
from app.config import Settings
from langchain_core.documents import Document

# --- Fixture for a service instance with mocked dependencies ---
@pytest.fixture
def mocked_ingestion_service(
    override_settings: Settings,
    mock_directory_loader: MagicMock,
    mock_text_splitter: MagicMock,
    mock_chroma_vector_store: MagicMock,
    mock_chroma_client: MagicMock,
    mocker
) -> IngestionProcessorService:
    """Provides an IngestionProcessorService with mocked internal components."""
    # Patch the dependency getters or the classes themselves if easier
    mocker.patch('app.services.ingestion_processor.DirectoryLoader', return_value=mock_directory_loader)
    mocker.patch('app.services.ingestion_processor.RecursiveCharacterTextSplitter', return_value=mock_text_splitter)
    # Patch the getters for Chroma/Embeddings to return mocks
    mocker.patch('app.services.ingestion_processor.get_vector_store', return_value=mock_chroma_vector_store)
    mocker.patch('app.services.ingestion_processor.get_chroma_client', return_value=mock_chroma_client)
    mocker.patch('app.services.ingestion_processor.get_embedding_model') # Don't need embedding mock directly if vector_store is mocked

    # Instantiate the service - it will now use the mocked components
    service = IngestionProcessorService(settings=override_settings)
    return service

# --- Unit Tests ---

def test_load_documents_success(
    mocked_ingestion_service: IngestionProcessorService,
    mock_directory_loader: MagicMock,
    override_settings: Settings,
    mocker
):
    """Test successful document loading."""
    # Mock Path.exists and is_dir to return True
    mocker.patch.object(Path, 'exists', return_value=True)
    mocker.patch.object(Path, 'is_dir', return_value=True)

    docs = mocked_ingestion_service._load_documents()

    mock_directory_loader.load.assert_called_once()
    assert len(docs) == 2 # Based on mock_directory_loader setup
    assert docs[0].page_content == "This is the first chunk of content."


def test_load_documents_dir_not_found(
    mocked_ingestion_service: IngestionProcessorService,
    override_settings: Settings,
    mocker
):
    """Test document loading when source directory doesn't exist."""
    mocker.patch.object(Path, 'exists', return_value=False) # Simulate dir not existing

    docs = mocked_ingestion_service._load_documents()
    assert docs == []


def test_split_documents(mocked_ingestion_service: IngestionProcessorService, mock_text_splitter: MagicMock):
    """Test document splitting logic."""
    input_docs = [
        Document(page_content="Long document content 1.", metadata={"source": "d1.pdf"}),
        Document(page_content="Long document content 2.", metadata={"source": "d2.pdf"}),
    ]
    chunks = mocked_ingestion_service._split_documents(input_docs)

    mock_text_splitter.split_documents.assert_called_once_with(input_docs)
    assert len(chunks) == 2 # Based on mock_text_splitter setup
    assert chunks[0].page_content == "Long document content 1."[:50] # Check content based on mock
    assert chunks[0].metadata["chunk"] == 0 # Check added metadata


def test_add_chunks_to_vector_store(
    mocked_ingestion_service: IngestionProcessorService,
    mock_chroma_vector_store: MagicMock
):
    """Test adding chunks to the mocked vector store."""
    input_chunks = [
        Document(page_content="chunk1", metadata={"source": "s1.pdf", "start_index": 0}),
        Document(page_content="chunk2", metadata={"source": "s1.pdf", "start_index": 100}),
        Document(page_content="chunk3", metadata={"source": "s2.pdf", "start_index": 0}),
    ]
    expected_ids = ["s1.pdf_chunk_0", "s1.pdf_chunk_100", "s2.pdf_chunk_0"]

    added_count = mocked_ingestion_service._add_chunks_to_vector_store(input_chunks)

    mock_chroma_vector_store.add_documents.assert_called_once_with(input_chunks, ids=expected_ids)
    assert added_count == 3


def test_add_chunks_failure(
    mocked_ingestion_service: IngestionProcessorService,
    mock_chroma_vector_store: MagicMock
):
    """Test handling failure during adding chunks."""
    mock_chroma_vector_store.add_documents.side_effect = Exception("DB Error")
    input_chunks = [Document(page_content="test")]

    added_count = mocked_ingestion_service._add_chunks_to_vector_store(input_chunks)

    mock_chroma_vector_store.add_documents.assert_called_once()
    assert added_count == 0


def test_run_ingestion_pipeline_success(
    mocked_ingestion_service: IngestionProcessorService,
    mock_directory_loader: MagicMock,
    mock_text_splitter: MagicMock,
    mock_chroma_vector_store: MagicMock,
    mocker
):
    """Test the full ingestion pipeline orchestration (success path)."""
    # Mock path checks
    mocker.patch.object(Path, 'exists', return_value=True)
    mocker.patch.object(Path, 'is_dir', return_value=True)

    status = mocked_ingestion_service.run_ingestion()

    # Assertions based on mocks
    mock_directory_loader.load.assert_called_once()
    mock_text_splitter.split_documents.assert_called_once()
    mock_chroma_vector_store.add_documents.assert_called_once()
    assert status.documents_processed == 2 # From mock loader
    assert status.chunks_added == 2 # From mock splitter -> add
    assert not status.errors


def test_run_ingestion_with_cleaning(
    mocked_ingestion_service: IngestionProcessorService,
    mock_chroma_client: MagicMock,
    mocker
):
    """Test the CLEAN_COLLECTION_BEFORE_INGEST flag."""
    # Modify settings for this test
    mocked_ingestion_service.settings.CLEAN_COLLECTION_BEFORE_INGEST = True
    # Mock path checks
    mocker.patch.object(Path, 'exists', return_value=True)
    mocker.patch.object(Path, 'is_dir', return_value=True)

    mocked_ingestion_service.run_ingestion()

    # Check if delete_collection was called on the underlying client mock
    mock_chroma_client.delete_collection.assert_called_once_with(
        mocked_ingestion_service.settings.CHROMA_COLLECTION_NAME
    )
    # Ensure other steps were still called
    assert mocked_ingestion_service.vector_store.add_documents.called


def test_run_ingestion_no_documents_loaded(
     mocked_ingestion_service: IngestionProcessorService,
     mock_directory_loader: MagicMock,
     mock_text_splitter: MagicMock,
     mock_chroma_vector_store: MagicMock,
     mocker
):
    """Test pipeline when no documents are loaded."""
    mocker.patch.object(Path, 'exists', return_value=True)
    mocker.patch.object(Path, 'is_dir', return_value=True)
    mock_directory_loader.load.return_value = [] # Simulate no docs loaded

    status = mocked_ingestion_service.run_ingestion()

    mock_directory_loader.load.assert_called_once()
    # Ensure split and add were NOT called
    mock_text_splitter.split_documents.assert_not_called()
    mock_chroma_vector_store.add_documents.assert_not_called()
    assert status.documents_processed == 0
    assert status.chunks_added == 0
    assert not status.errors

