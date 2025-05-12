import logging
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from app.config import Settings
from app.models import IngestionStatus
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.document_loaders import (
#    DirectoryLoader,
# )
# from langchain_unstructured import UnstructuredLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global instances (consider managing via lifespan/dependency injection if service runs long) ---
_embedding_model: Optional[SentenceTransformerEmbeddings] = None
_chroma_client: Optional[chromadb.ClientAPI] = None
_vector_store: Optional[Chroma] = None


def get_chroma_client(settings: Settings) -> chromadb.ClientAPI:
    """Gets or creates a ChromaDB client based on settings."""
    global _chroma_client
    if _chroma_client is None:
        chroma_mode = settings.CHROMA_MODE
        if chroma_mode == "local":
            chroma_path = settings.CHROMA_PATH
            print(f"DEBUG: Connecting to local ChromaDB at path: {chroma_path}")
            print(f"ChromaDB path: {chroma_path}")
            if not chroma_path:
                raise ValueError("chroma_path is required for local mode.")
            _chroma_client = chromadb.PersistentClient(path=chroma_path)

        elif chroma_mode == "docker":
            chroma_host = settings.CHROMA_HOST
            print(
                f"DEBUG: Connecting to ChromaDB Docker container at host: {chroma_host}"
            )
            if not chroma_host:
                raise ValueError("chroma_host is required for docker mode.")
            print(f"ChromaDB host: {chroma_host}")
            chroma_port = settings.CHROMA_PORT
            print(
                f"DEBUG: Connecting to ChromaDB Docker container at port: {chroma_port}"
            )
            if not chroma_port:
                raise ValueError("chroma_port is required for docker mode.")
            _chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            raise ValueError(
                f"Invalid CHROMA_MODE: {chroma_mode}. Must be 'local' or 'docker'."
            )

    return _chroma_client


def get_embedding_model(settings: Settings) -> SentenceTransformerEmbeddings:
    """Gets or creates the embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        try:
            _embedding_model = SentenceTransformerEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            _embedding_model = None
            raise RuntimeError(f"Failed to load embedding model: {e}") from e
    return _embedding_model


def get_vector_store(settings: Settings) -> Chroma:
    """Gets or creates the LangChain Chroma vector store instance."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing LangChain Chroma vector store...")
        client = get_chroma_client(settings)
        embedding_function = get_embedding_model(settings)

        try:
            _vector_store = Chroma(
                client=client,
                collection_name=settings.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_function,
            )
            logger.info(
                f"LangChain Chroma vector store connected to collection '{settings.CHROMA_COLLECTION_NAME}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize LangChain Chroma vector store: {e}",
                exc_info=True,
            )
            _vector_store = None
            raise RuntimeError(
                f"Failed to initialize LangChain Chroma vector store: {e}"
            ) from e
    return _vector_store


class IngestionProcessorService:
    """Handles the document ingestion process."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.source_directory = Path(settings.SOURCE_DIRECTORY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        self.vector_store = get_vector_store(settings)
        logger.info("IngestionProcessorService initialized.")

    def _load_documents2(self) -> List[Document]:
        """Loads PDF documents from the source directory."""
        if not self.source_directory.exists() or not self.source_directory.is_dir():
            logger.error(
                f"Source directory not found or is not a directory: {self.source_directory}"
            )
            return []

        logger.info(f"Loading PDF documents from: {self.source_directory}")

        # Configure DirectoryLoader to load PDF files using UnstructuredFileLoader
        loader = DirectoryLoader(
            path=str(self.source_directory),
            glob="**/*.pdf",  # Pattern to find only PDF files
            loader_cls=UnstructuredLoader,
            use_multithreading=True,
            show_progress=True,
            recursive=True,
            silent_errors=True,  # Log errors but attempt to continue
        )

        try:
            documents = loader.load()
            logger.info(
                f"Attempted loading PDFs. Found {len(documents)} document objects."
            )
            # Basic filtering: remove documents with empty page_content
            loaded_docs = [
                doc
                for doc in documents
                if doc.page_content and doc.page_content.strip()
            ]
            if len(loaded_docs) < len(documents):
                logger.warning(
                    f"Removed {len(documents) - len(loaded_docs)} documents with empty or whitespace-only content after loading."
                )
            if not loaded_docs:
                logger.warning(
                    "No valid content could be extracted from the PDF files found."
                )
            return loaded_docs
        except Exception as e:
            logger.error(f"Error loading PDF documents: {e}", exc_info=True)
            return []  # Return empty list on failure

    def _load_documents3(self) -> List[Document]:
        logger.info("Attempting to load a single test PDF.")
        # Replace with an actual path to a test PDF
        test_pdf_path = r"app\documents\iphone-16-info.pdf"
        if not Path(test_pdf_path).exists():
            logger.error(f"Test PDF not found at: {test_pdf_path}")
            return []

        try:
            loader = PyPDFLoader(test_pdf_path)
            logger.info(f"Loading single test PDF: {test_pdf_path}")
            documents = loader.load()
            # Filter out documents with no content or only whitespace
            valid_documents = [
                doc
                for doc in documents
                if doc.page_content and doc.page_content.strip()
            ]
            logger.info(
                f"Loaded {len(valid_documents)} valid document(s) from the single test PDF."
            )
            return valid_documents
        except Exception as e:
            logger.error(
                f"Error loading single test PDF {test_pdf_path}: {e}", exc_info=True
            )
            return []

    def _load_documents(self) -> List[Document]:
        """Loads all PDF documents from the source directory and its subdirectories."""
        if not self.source_directory.exists() or not self.source_directory.is_dir():
            logger.error(
                f"Source directory not found or is not a directory: {self.source_directory}"
            )
            return []

        logger.info(f"Loading all PDF documents from: {self.source_directory}")

        all_documents: List[Document] = []
        pdf_files_found = list(self.source_directory.rglob("*.pdf")) # Recursively find all .pdf files

        if not pdf_files_found:
            logger.warning(f"No PDF files found in {self.source_directory}")
            return []

        logger.info(f"Found {len(pdf_files_found)} PDF files to process.")

        for pdf_path in pdf_files_found:
            try:
                logger.info(f"Loading PDF: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                documents_from_file = loader.load()

                # Filter out documents with no content or only whitespace
                valid_documents_from_file = [
                    doc
                    for doc in documents_from_file
                    if doc.page_content and doc.page_content.strip()
                ]

                if valid_documents_from_file:
                    all_documents.extend(valid_documents_from_file)
                    logger.info(
                        f"Loaded {len(valid_documents_from_file)} valid document(s) from {pdf_path}."
                    )
                else:
                    logger.warning(f"No valid content extracted from {pdf_path}.")
            except Exception as e:
                logger.error(
                    f"Error loading PDF document {pdf_path}: {e}", exc_info=True
                )
                # Optionally, continue to try loading other PDFs

        if not all_documents:
            logger.warning(
                "No valid content could be extracted from any of the PDF files found."
            )
        else:
            logger.info(
                f"Successfully loaded a total of {len(all_documents)} valid document(s) from all PDF files."
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
        """Adds document chunks to the Chroma vector store."""
        if not chunks:
            logger.warning("No chunks to add to the vector store.")
            return 0

        logger.info(
            f"Adding {len(chunks)} chunks to Chroma collection '{self.settings.CHROMA_COLLECTION_NAME}'..."
        )

        try:
            # Generate more unique IDs based on source, page number, and chunk start index
            ids = []
            for i, chunk in enumerate(chunks):
                source_name = os.path.basename(
                    chunk.metadata.get("source", f"unknown_source_{i}")
                )
                page_number = chunk.metadata.get(
                    "page", 0
                )  # Get page number from metadata
                start_index = chunk.metadata.get(
                    "start_index", i
                )  # Get start_index from metadata
                ids.append(f"{source_name}_page_{page_number}_chunk_{start_index}")

            self.vector_store.add_documents(chunks, ids=ids)
            logger.info(f"Successfully added {len(chunks)} chunks to the vector store.")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}", exc_info=True)
            return 0

    def run_ingestion(self) -> IngestionStatus:
        """Executes the full ingestion pipeline."""
        status = IngestionStatus()
        logger.info("Starting ingestion process...")

        # Optional: Clean collection
        if self.settings.CLEAN_COLLECTION_BEFORE_INGEST:
            logger.warning(
                f"CLEAN_COLLECTION_BEFORE_INGEST is True. Deleting collection '{self.settings.CHROMA_COLLECTION_NAME}'..."
            )
            try:
                client = get_chroma_client(self.settings)
                client.delete_collection(self.settings.CHROMA_COLLECTION_NAME)
                logger.info(
                    f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' deleted."
                )
                global _vector_store
                _vector_store = None  # Reset store instance
                self.vector_store = get_vector_store(self.settings)  # Re-initialize
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.info(
                        f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' did not exist, no need to delete."
                    )
                else:
                    logger.error(
                        f"Failed to delete collection '{self.settings.CHROMA_COLLECTION_NAME}': {e}",
                        exc_info=True,
                    )
                    status.errors.append(f"Failed to delete collection: {e}")
                    # return status # Optional: Stop if cleaning failed

        # Load PDF documents
        documents = self._load_documents()
        status.documents_processed = len(
            documents
        )  # Count documents with actual content
        if not documents:
            logger.warning("No valid document content loaded, ingestion finished.")
            return status

        # Split documents
        chunks = self._split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")
        if not chunks:
            logger.warning("No chunks created after splitting, ingestion finished.")
            return status

        # Embed and Store chunks
        added_count = self._add_chunks_to_vector_store(chunks)
        status.chunks_added = added_count
        if added_count < len(chunks):
            status.errors.append(
                "Failed to add some or all chunks to the vector store."
            )

        logger.info(
            f"Ingestion process finished. Processed: {status.documents_processed} docs, Added: {status.chunks_added} chunks."
        )
        if status.errors:
            logger.error(f"Ingestion completed with errors: {status.errors}")

        return status
