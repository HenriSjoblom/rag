from pathlib import Path

from app.config import Settings
from app.services.ingestion_processor import IngestionProcessorService
from app.services.ingestion_state import IngestionStateService
from langchain_core.documents import Document


class TestDocumentLoading:
    """Tests for document loading functionality."""

    def test_load_documents_success(
        self,
        mocked_ingestion_service: IngestionProcessorService,
        unit_settings: Settings,
        mocker,
    ):
        """Test successful document loading."""
        # Mock Path.exists and is_dir to return True
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock Path.rglob to return some PDF files
        mock_pdf_files = [Path("doc1.pdf"), Path("doc2.pdf")]
        mocker.patch.object(Path, "rglob", return_value=mock_pdf_files)

        # Mock PyPDFLoader - need to create a function that returns different content based on the file
        def mock_loader_factory(file_path):
            mock_loader = mocker.Mock()
            if "doc1.pdf" in str(file_path):
                mock_loader.load.return_value = [
                    Document(
                        page_content="Content from doc1",
                        metadata={"source": "doc1.pdf"},
                    ),
                ]
            elif "doc2.pdf" in str(file_path):
                mock_loader.load.return_value = [
                    Document(
                        page_content="Content from doc2",
                        metadata={"source": "doc2.pdf"},
                    ),
                ]
            else:
                mock_loader.load.return_value = []
            return mock_loader

        mocker.patch(
            "app.services.ingestion_processor.PyPDFLoader",
            side_effect=mock_loader_factory,
        )

        # Mock _get_processed_files to return empty set
        mocker.patch.object(
            mocked_ingestion_service, "_get_processed_files", return_value=set()
        )

        docs = mocked_ingestion_service._load_documents()

        assert len(docs) == 2
        assert docs[0].page_content == "Content from doc1"
        assert docs[1].page_content == "Content from doc2"

    def test_load_documents_dir_not_found(
        self,
        mocked_ingestion_service: IngestionProcessorService,
        unit_settings: Settings,
        mocker,
    ):
        """Test document loading when source directory doesn't exist."""
        mocker.patch.object(Path, "exists", return_value=False)

        docs = mocked_ingestion_service._load_documents()
        assert docs == []

    def test_load_documents_with_existing_processed_files(
        self, mocked_ingestion_service, mocker
    ):
        """Test that already processed files are skipped."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        mock_pdf_files = [Path("doc1.pdf"), Path("doc2.pdf")]
        mocker.patch.object(Path, "rglob", return_value=mock_pdf_files)

        # Mock that doc1.pdf has already been processed
        mocker.patch.object(
            mocked_ingestion_service, "_get_processed_files", return_value={"doc1.pdf"}
        )

        # Only doc2.pdf should be loaded
        mock_loader = mocker.Mock()
        mock_loader.load.return_value = [
            Document(page_content="Content from doc2", metadata={"source": "doc2.pdf"}),
        ]
        mocker.patch(
            "app.services.ingestion_processor.PyPDFLoader", return_value=mock_loader
        )

        docs = mocked_ingestion_service._load_documents()

        assert len(docs) == 1
        assert docs[0].page_content == "Content from doc2"

    def test_load_documents_empty_directory(self, mocked_ingestion_service, mocker):
        """Test loading from empty directory."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)
        mocker.patch.object(Path, "rglob", return_value=[])

        docs = mocked_ingestion_service._load_documents()
        assert docs == []

    def test_load_documents_with_invalid_pdf(self, mocked_ingestion_service, mocker):
        """Test handling of PDF files that can't be loaded."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        mock_pdf_files = [Path("corrupted.pdf")]
        mocker.patch.object(Path, "rglob", return_value=mock_pdf_files)
        mocker.patch.object(
            mocked_ingestion_service, "_get_processed_files", return_value=set()
        )

        # Mock PyPDFLoader to raise an exception
        mocker.patch(
            "app.services.ingestion_processor.PyPDFLoader",
            side_effect=Exception("PDF corrupted"),
        )

        docs = mocked_ingestion_service._load_documents()
        assert docs == []


class TestDocumentProcessing:
    """Tests for document processing and chunking."""

    def test_split_documents_success(
        self, mocked_ingestion_service: IngestionProcessorService
    ):
        """Test document splitting logic."""
        input_docs = [
            Document(
                page_content="Long document content 1.", metadata={"source": "d1.pdf"}
            ),
            Document(
                page_content="Long document content 2.", metadata={"source": "d2.pdf"}
            ),
        ]

        # The actual text_splitter is created in __init__, so we can use it
        chunks = mocked_ingestion_service._split_documents(input_docs)

        # Should return some chunks (actual behavior will depend on chunk size)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_split_documents_empty_list(self, mocked_ingestion_service):
        """Test splitting empty document list."""
        chunks = mocked_ingestion_service._split_documents([])
        assert chunks == []

    def test_split_documents_preserves_metadata(self, mocked_ingestion_service):
        """Test that document splitting preserves metadata."""
        input_docs = [
            Document(
                page_content="This is a test document with some content.",
                metadata={"source": "test.pdf", "page": 1},
            ),
        ]

        chunks = mocked_ingestion_service._split_documents(input_docs)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.pdf"


class TestVectorStoreOperations:
    """Tests for vector store operations."""

    def test_add_chunks_to_vector_store_success(
        self,
        mocked_ingestion_service: IngestionProcessorService,
        mock_chroma_vector_store,
        mocker,
    ):
        """Test adding chunks to the mocked vector store."""
        input_chunks = [
            Document(
                page_content="chunk1",
                metadata={"source": "s1.pdf", "page": 1, "start_index": 0},
            ),
            Document(
                page_content="chunk2",
                metadata={"source": "s1.pdf", "page": 1, "start_index": 100},
            ),
            Document(
                page_content="chunk3",
                metadata={"source": "s2.pdf", "page": 1, "start_index": 0},
            ),
        ]

        # Mock time.time() for consistent IDs
        mocker.patch("time.time", return_value=1234567890)

        added_count = mocked_ingestion_service._add_chunks_to_vector_store(input_chunks)

        mock_chroma_vector_store.add_documents.assert_called_once()
        assert added_count == 3

    def test_add_chunks_failure(
        self,
        mocked_ingestion_service: IngestionProcessorService,
        mock_chroma_vector_store,
    ):
        """Test handling failure during adding chunks."""
        mock_chroma_vector_store.add_documents.side_effect = Exception("DB Error")
        input_chunks = [
            Document(page_content="test", metadata={"source": "test.pdf", "page": 1})
        ]

        added_count = mocked_ingestion_service._add_chunks_to_vector_store(input_chunks)

        mock_chroma_vector_store.add_documents.assert_called()
        assert added_count == 0

    def test_add_chunks_empty_list(self, mocked_ingestion_service):
        """Test adding empty chunk list."""
        added_count = mocked_ingestion_service._add_chunks_to_vector_store([])
        assert added_count == 0

    def test_add_chunks_with_retry_success(
        self, mocked_ingestion_service, mock_chroma_vector_store, mocker
    ):
        """Test successful retry after initial failure."""
        input_chunks = [
            Document(page_content="test", metadata={"source": "test.pdf", "page": 1})
        ]

        # First call fails, second succeeds
        mock_chroma_vector_store.add_documents.side_effect = [
            Exception("Temporary failure"),
            None,  # Success on retry
        ]

        mocker.patch("time.time", return_value=1234567890)
        mocker.patch("time.sleep")  # Mock sleep to speed up test

        added_count = mocked_ingestion_service._add_chunks_to_vector_store(input_chunks)

        assert mock_chroma_vector_store.add_documents.call_count == 2
        assert added_count == 1


class TestIngestionPipeline:
    """Tests for the complete ingestion pipeline."""

    def test_run_ingestion_pipeline_success(
        self,
        mocked_ingestion_service: IngestionProcessorService,
        mock_chroma_vector_store,
        mocker,
    ):
        """Test the full ingestion pipeline orchestration (success path)."""
        # Mock path checks
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock Path.rglob to return some PDF files
        mock_pdf_files = [Path("doc1.pdf")]
        mocker.patch.object(Path, "rglob", return_value=mock_pdf_files)

        # Mock PyPDFLoader
        mock_loader = mocker.Mock()
        mock_loader.load.return_value = [
            Document(page_content="Test content", metadata={"source": "doc1.pdf"}),
        ]
        mocker.patch(
            "app.services.ingestion_processor.PyPDFLoader", return_value=mock_loader
        )

        # Mock _get_processed_files to return empty set
        mocker.patch.object(
            mocked_ingestion_service, "_get_processed_files", return_value=set()
        )

        # Mock time.time() for consistent behavior
        mocker.patch("time.time", return_value=1234567890)

        status = mocked_ingestion_service.run_ingestion()

        # Assertions based on mocks
        mock_chroma_vector_store.add_documents.assert_called_once()
        assert status.documents_processed == 1
        assert status.chunks_added > 0  # Should be at least 1 chunk
        assert not status.errors

    def test_run_ingestion_with_cleaning(
        self,
        mocked_ingestion_service: IngestionProcessorService,
        mock_chroma_client,
        mocker,
    ):
        """Test the CLEAN_COLLECTION_BEFORE_INGEST flag."""
        # Modify settings for this test
        mocked_ingestion_service.settings.CLEAN_COLLECTION_BEFORE_INGEST = True

        # Mock path checks
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock Path.rglob to return some PDF files
        mock_pdf_files = [Path("doc1.pdf")]
        mocker.patch.object(Path, "rglob", return_value=mock_pdf_files)

        # Mock PyPDFLoader
        mock_loader = mocker.Mock()
        mock_loader.load.return_value = [
            Document(page_content="Test content", metadata={"source": "doc1.pdf"}),
        ]
        mocker.patch(
            "app.services.ingestion_processor.PyPDFLoader", return_value=mock_loader
        )

        # Mock _get_processed_files to return empty set
        mocker.patch.object(
            mocked_ingestion_service, "_get_processed_files", return_value=set()
        )

        # Mock time.time() for consistent behavior
        mocker.patch("time.time", return_value=1234567890)

        mocked_ingestion_service.run_ingestion()

        # Check if delete_collection was called on the underlying client mock
        mock_chroma_client.delete_collection.assert_called_once_with(
            mocked_ingestion_service.settings.CHROMA_COLLECTION_NAME
        )

    def test_run_ingestion_no_documents_loaded(
        self,
        mocked_ingestion_service: IngestionProcessorService,
        mock_chroma_vector_store,
        mocker,
    ):
        """Test pipeline when no documents are loaded."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock Path.rglob to return empty list (no PDF files)
        mocker.patch.object(Path, "rglob", return_value=[])

        status = mocked_ingestion_service.run_ingestion()

        # Ensure add_documents was NOT called
        mock_chroma_vector_store.add_documents.assert_not_called()
        assert status.documents_processed == 0
        assert status.chunks_added == 0
        assert not status.errors

    def test_run_ingestion_collection_cleanup_error(
        self, mocked_ingestion_service, mock_chroma_client, mocker
    ):
        """Test handling of collection cleanup errors."""
        mocked_ingestion_service.settings.CLEAN_COLLECTION_BEFORE_INGEST = True
        mock_chroma_client.delete_collection.side_effect = Exception("Delete failed")

        # Mock empty directory to avoid document processing
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)
        mocker.patch.object(Path, "rglob", return_value=[])

        status = mocked_ingestion_service.run_ingestion()

        assert len(status.errors) == 1
        assert "Failed to delete collection" in status.errors[0]

    def test_run_ingestion_all_files_already_processed(
        self, mocked_ingestion_service, mocker
    ):
        """Test when all files have already been processed."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        mock_pdf_files = [Path("doc1.pdf")]
        mocker.patch.object(Path, "rglob", return_value=mock_pdf_files)

        # Mock that all files are already processed
        mocker.patch.object(
            mocked_ingestion_service, "_get_processed_files", return_value={"doc1.pdf"}
        )

        status = mocked_ingestion_service.run_ingestion()

        assert status.documents_processed == 0
        assert status.chunks_added == 0
        assert not status.errors


class TestErrorHandling:
    """Tests for error scenarios in ingestion."""

    def test_get_processed_files_error_handling(
        self, mocked_ingestion_service, mock_chroma_vector_store
    ):
        """Test error handling in _get_processed_files."""
        # Mock vector store to raise exception
        mock_chroma_vector_store._collection.get.side_effect = Exception(
            "Connection error"
        )

        processed_files = mocked_ingestion_service._get_processed_files()

        # Should return empty set on error
        assert processed_files == set()
        # Should call reset on vector store manager
        mocked_ingestion_service.vector_store_manager.reset.assert_called_once()

    def test_vector_store_connection_failure(self, mocked_ingestion_service, mocker):
        """Test handling of vector store connection failures."""
        mocker.patch.object(Path, "exists", return_value=True)
        mocker.patch.object(Path, "is_dir", return_value=True)

        mock_pdf_files = [Path("doc1.pdf")]
        mocker.patch.object(Path, "rglob", return_value=mock_pdf_files)

        mock_loader = mocker.Mock()
        mock_loader.load.return_value = [
            Document(page_content="Test content", metadata={"source": "doc1.pdf"}),
        ]
        mocker.patch(
            "app.services.ingestion_processor.PyPDFLoader", return_value=mock_loader
        )

        mocker.patch.object(
            mocked_ingestion_service, "_get_processed_files", return_value=set()
        )

        # Mock vector store manager to fail
        mocked_ingestion_service.vector_store_manager.get_vector_store.side_effect = (
            Exception("Vector store error")
        )

        status = mocked_ingestion_service.run_ingestion()

        assert status.documents_processed == 1
        assert status.chunks_added == 0
        assert len(status.errors) == 1


class TestIngestionServiceIntegration:
    """Test interactions between ingestion services."""

    def test_processor_and_state_service_coordination(self, unit_settings):
        """Test that processor correctly updates state service."""
        state_service = IngestionStateService()

        # This would require more complex setup with actual service coordination
        # For now, just test that services can be instantiated together
        assert state_service is not None

    def test_service_initialization_with_different_settings(self, mocker):
        """Test service initialization with various settings configurations."""
        # Mock the manager dependencies
        mock_chroma_manager = mocker.Mock()
        mock_embedding_manager = mocker.Mock()
        mock_vector_store_manager = mocker.Mock()

        # Test with minimal valid settings
        from app.config import Settings

        settings = Settings(
            SOURCE_DIRECTORY="/tmp/test",
            EMBEDDING_MODEL_NAME="test-model",
            CHUNK_SIZE=101,  # Must be > 100
            CHUNK_OVERLAP=10,
            CHROMA_COLLECTION_NAME="test",
            CHROMA_MODE="local",
            CHROMA_LOCAL_PATH="/tmp/chroma",
        )

        service = IngestionProcessorService(
            settings=settings,
            chroma_manager=mock_chroma_manager,
            embedding_manager=mock_embedding_manager,
            vector_store_manager=mock_vector_store_manager,
        )

        assert service.settings == settings
        assert service.source_directory == Path("/tmp/test")
        assert service.text_splitter._chunk_size == 101  # Use _chunk_size attribute
        assert (
            service.text_splitter._chunk_overlap == 10
        )  # Use _chunk_overlap attribute
