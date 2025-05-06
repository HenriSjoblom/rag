import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb

from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader
)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import Settings
from app.models import IngestionStatus


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
        if settings.CHROMA_PATH:
            logger.info(f"Initializing persistent ChromaDB client at path: {settings.CHROMA_PATH}")
            try:
                # Pass the path directly to the client constructor
                _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
                logger.info("Persistent ChromaDB client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize persistent ChromaDB client at {settings.CHROMA_PATH}: {e}", exc_info=True)
                _chroma_client = None # Ensure it's None if failed
                raise RuntimeError(f"Failed to initialize ChromaDB client: {e}") from e
        else:
           raise ValueError("CHROMA_PATH is not set in settings. Cannot initialize ChromaDB client.")

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
            logger.info(f"LangChain Chroma vector store connected to collection '{settings.CHROMA_COLLECTION_NAME}'.")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Chroma vector store: {e}", exc_info=True)
            _vector_store = None
            raise RuntimeError(f"Failed to initialize LangChain Chroma vector store: {e}") from e
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

    def _load_documents(self) -> List[Document]:
        """Loads PDF documents from the source directory."""
        if not self.source_directory.exists() or not self.source_directory.is_dir():
             logger.error(f"Source directory not found or is not a directory: {self.source_directory}")
             return []

        logger.info(f"Loading PDF documents from: {self.source_directory}")

        # Configure DirectoryLoader to load PDF files using UnstructuredFileLoader
        loader = DirectoryLoader(
            path=str(self.source_directory),
            glob="**/*.pdf",  # Pattern to find only PDF files
            loader_cls=UnstructuredFileLoader,
            use_multithreading=True,
            show_progress=True,
            recursive=True,
            silent_errors=True # Log errors but attempt to continue
        )

        try:
            documents = loader.load()
            logger.info(f"Attempted loading PDFs. Found {len(documents)} document objects.")
            # Basic filtering: remove documents with empty page_content
            loaded_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
            if len(loaded_docs) < len(documents):
                 logger.warning(f"Removed {len(documents) - len(loaded_docs)} documents with empty or whitespace-only content after loading.")
            if not loaded_docs:
                 logger.warning("No valid content could be extracted from the PDF files found.")
            return loaded_docs
        except Exception as e:
            logger.error(f"Error loading PDF documents: {e}", exc_info=True)
            return [] # Return empty list on failure

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

        logger.info(f"Adding {len(chunks)} chunks to Chroma collection '{self.settings.CHROMA_COLLECTION_NAME}'...")

        try:
            # Generate simple IDs based on source filename and chunk index
            ids = []
            for i, chunk in enumerate(chunks):
                 source_name = os.path.basename(chunk.metadata.get('source', f'unknown_source_{i}'))
                 # Ensure ID uniqueness if multiple chunks come from the same source index (though less likely with add_start_index)
                 start_index = chunk.metadata.get('start_index', i)
                 ids.append(f"{source_name}_chunk_{start_index}")

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
            logger.warning(f"CLEAN_COLLECTION_BEFORE_INGEST is True. Deleting collection '{self.settings.CHROMA_COLLECTION_NAME}'...")
            try:
                client = get_chroma_client(self.settings)
                client.delete_collection(self.settings.CHROMA_COLLECTION_NAME)
                logger.info(f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' deleted.")
                global _vector_store
                _vector_store = None # Reset store instance
                self.vector_store = get_vector_store(self.settings) # Re-initialize
            except Exception as e:
                 if "does not exist" in str(e).lower():
                      logger.info(f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' did not exist, no need to delete.")
                 else:
                      logger.error(f"Failed to delete collection '{self.settings.CHROMA_COLLECTION_NAME}': {e}", exc_info=True)
                      status.errors.append(f"Failed to delete collection: {e}")
                      # return status # Optional: Stop if cleaning failed

        # Load PDF documents
        documents = self._load_documents()
        status.documents_processed = len(documents) # Count documents with actual content
        if not documents:
            logger.warning("No valid document content loaded, ingestion finished.")
            return status

        # Split documents
        chunks = self._split_documents(documents)
        if not chunks:
            logger.warning("No chunks created after splitting, ingestion finished.")
            return status

        # Embed and Store chunks
        added_count = self._add_chunks_to_vector_store(chunks)
        status.chunks_added = added_count
        if added_count < len(chunks):
            status.errors.append("Failed to add some or all chunks to the vector store.")

        logger.info(f"Ingestion process finished. Processed: {status.documents_processed} docs, Added: {status.chunks_added} chunks.")
        if status.errors:
             logger.error(f"Ingestion completed with errors: {status.errors}")

        return status
