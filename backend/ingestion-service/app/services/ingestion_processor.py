import logging
import os
from pathlib import Path
from typing import List

from app.config import Settings
from app.models import IngestionStatus
from app.services.chroma_manager import (
    ChromaClientManager,
    EmbeddingModelManager,
    VectorStoreManager,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class IngestionProcessorService:
    """Handles the document ingestion process."""

    def __init__(
        self,
        settings: Settings,
        chroma_manager: ChromaClientManager,
        embedding_manager: EmbeddingModelManager,
        vector_store_manager: VectorStoreManager,
    ):
        self.settings = settings
        self.source_directory = Path(settings.SOURCE_DIRECTORY)
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        logger.info("IngestionProcessorService initialized.")

    def _get_processed_files(self) -> set:
        """Get a set of already processed file names from the vector store."""
        try:
            vector_store = self.vector_store_manager.get_vector_store()
            collection = vector_store._collection
            all_docs = collection.get()

            processed_files = set()
            if all_docs and "metadatas" in all_docs:
                for metadata in all_docs["metadatas"]:
                    if metadata and "source" in metadata:
                        source_path = Path(metadata["source"])
                        processed_files.add(source_path.name)

            logger.info(
                f"Found {len(processed_files)} already processed files: {processed_files}"
            )
            return processed_files
        except Exception as e:
            logger.warning(f"Could not retrieve processed files list: {e}")
            self.vector_store_manager.reset()
            return set()

    def _load_documents(self) -> List[Document]:
        """Loads only new PDF documents that haven't been processed yet."""
        if not self.source_directory.exists() or not self.source_directory.is_dir():
            logger.error(f"Source directory not found: {self.source_directory}")
            return []

        logger.info(f"Loading PDF documents from: {self.source_directory}")

        processed_files = self._get_processed_files()
        all_documents: List[Document] = []
        pdf_files_found = list(self.source_directory.rglob("*.pdf"))

        if not pdf_files_found:
            logger.warning(f"No PDF files found in {self.source_directory}")
            return []

        new_pdf_files = [f for f in pdf_files_found if f.name not in processed_files]

        if not new_pdf_files:
            logger.info(
                "All PDF files have already been processed. No new files to ingest."
            )
            return []

        logger.info(
            f"Found {len(pdf_files_found)} total PDFs, {len(new_pdf_files)} new files to process."
        )

        for pdf_path in new_pdf_files:
            try:
                logger.info(f"Loading new PDF: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                documents_from_file = loader.load()

                valid_documents = [
                    doc
                    for doc in documents_from_file
                    if doc.page_content and doc.page_content.strip()
                ]

                if valid_documents:
                    all_documents.extend(valid_documents)
                    logger.info(
                        f"Loaded {len(valid_documents)} valid pages from {pdf_path}"
                    )
                else:
                    logger.warning(f"No valid content extracted from {pdf_path}")
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_path}: {e}", exc_info=True)

        logger.info(
            f"Successfully loaded {len(all_documents)} valid pages from {len(new_pdf_files)} new PDFs."
        )
        return all_documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits loaded documents into smaller chunks."""
        if not documents:
            return []
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")
        return chunks

    def _add_chunks_to_vector_store(self, chunks: List[Document]) -> int:
        """Adds document chunks to the Chroma vector store with retry logic."""
        if not chunks:
            logger.warning("No chunks to add to the vector store.")
            return 0

        logger.info(
            f"Adding {len(chunks)} chunks to collection '{self.settings.CHROMA_COLLECTION_NAME}'..."
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}")
                    self.vector_store_manager.reset()

                vector_store = self.vector_store_manager.get_vector_store()

                # Generate unique IDs with timestamp
                import time

                timestamp = int(time.time())
                ids = []
                for i, chunk in enumerate(chunks):
                    source_name = os.path.basename(
                        chunk.metadata.get("source", f"unknown_{i}")
                    )
                    page_number = chunk.metadata.get("page", 0)
                    start_index = chunk.metadata.get("start_index", i)
                    ids.append(
                        f"{source_name}_p{page_number}_c{start_index}_{timestamp}"
                    )

                vector_store.add_documents(chunks, ids=ids)
                logger.info(
                    f"Successfully added {len(chunks)} chunks to the vector store."
                )
                return len(chunks)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to add chunks after {max_retries} attempts")
                    return 0
                import time

                time.sleep(2**attempt)  # Exponential backoff

        return 0

    def run_ingestion(self) -> IngestionStatus:  # Sync method
        """Executes the full ingestion pipeline."""
        status = IngestionStatus()
        logger.info("Starting ingestion process...")

        # Optional collection cleanup
        if self.settings.CLEAN_COLLECTION_BEFORE_INGEST:
            logger.warning(
                f"Cleaning collection '{self.settings.CHROMA_COLLECTION_NAME}'..."
            )
            try:
                client = self.chroma_manager.get_client()
                client.delete_collection(self.settings.CHROMA_COLLECTION_NAME)
                logger.info(
                    f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' deleted."
                )
                self.vector_store_manager.reset()
            except Exception as e:
                if "does not exist" not in str(e).lower():
                    logger.error(f"Failed to delete collection: {e}", exc_info=True)
                    status.errors.append(f"Failed to delete collection: {e}")

        # Load and process documents
        documents = self._load_documents()
        status.documents_processed = len(documents)

        if not documents:
            logger.warning("No documents loaded, ingestion finished.")
            return status

        chunks = self._split_documents(documents)
        if not chunks:
            logger.warning("No chunks created, ingestion finished.")
            return status

        added_count = self._add_chunks_to_vector_store(chunks)
        status.chunks_added = added_count

        if added_count < len(chunks):
            status.errors.append("Failed to add some chunks to the vector store.")

        logger.info(
            f"Ingestion completed. Documents: {status.documents_processed}, Chunks: {status.chunks_added}"
        )
        return status
