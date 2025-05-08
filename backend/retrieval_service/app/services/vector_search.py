import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global instances managed by lifespan
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None
_chroma_collection: Optional[chromadb.Collection] = None


async def get_embedding_model() -> SentenceTransformer:
    """Dependency to get the loaded embedding model."""
    if _embedding_model is None:
        raise RuntimeError("Embedding model not initialized.")
    return _embedding_model


async def get_chroma_collection() -> chromadb.Collection:
    """Dependency to get the ChromaDB collection object."""
    if _chroma_collection is None:
        raise RuntimeError("ChromaDB collection not initialized.")
    return _chroma_collection


@asynccontextmanager
async def lifespan_retrieval_service(
    app,
    model_name: str,
    chroma_mode: str,
    collection_name: str,
    chroma_path: Optional[str] = None,
    chroma_host: Optional[str] = None,
):
    """Manages the lifespan of embedding model and ChromaDB client."""
    global _embedding_model, _chroma_client, _chroma_collection

    # --- For testing ---
    if _embedding_model is not None and _chroma_collection is not None:
        print(
            "Lifespan: Resources already initialized (idempotency check). Skipping setup."
        )
        try:
            yield
        finally:
            pass  # No teardown needed if setup was skipped
        return  # Exit early

    # Load Model
    print("Loading embedding model...")
    print(f"Collection name: {collection_name}")
    print(f"ChromaDB mode: {chroma_mode}")
    logger.info(f"Loading embedding model: {model_name}...")
    try:
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(
            f"Failed to load embedding model '{model_name}': {e}", exc_info=True
        )
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

    # Connect to ChromaDB
    # Validate chroma_mode and required parameters
    if chroma_mode == "local":
        print(f"DEBUG: Connecting to local ChromaDB at path: {chroma_path}")
        print(f"ChromaDB path: {chroma_path}")
        if not chroma_path:
            raise ValueError("chroma_path is required for local mode.")
        _chroma_client = chromadb.PersistentClient(
            path=chroma_path, settings=ChromaSettings(allow_reset=True)
        )
    elif chroma_mode == "docker":
        print(f"DEBUG: Connecting to ChromaDB Docker container at host: {chroma_host}")
        if not chroma_host:
            raise ValueError("chroma_host is required for docker mode.")
        print(f"ChromaDB host: {chroma_host}")
        _chroma_client = chromadb.HttpClient(host=chroma_host)
    else:
        raise ValueError(f"Invalid CHROMA_MODE: {chroma_mode}. Must be 'local' or 'docker'.")
    print("ChromaDB client initialized.")
    try:
        print(f"DEBUG: Creating or retrieving ChromaDB collection '{collection_name}'...")
        _chroma_collection = _chroma_client.get_or_create_collection(name=collection_name)
        print(f"DEBUG: ChromaDB collection '{_chroma_collection.name}' initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize ChromaDB collection: {e}")
        raise RuntimeError(f"Failed to initialize ChromaDB collection: {e}")

    try:
        yield  # Application runs
    finally:
        # Cleanup
        logger.info("Shutting down retrieval service resources...")
        if _chroma_client:
            try:
                print("Resetting ChromaDB client...")
                _chroma_client.reset()  # Release file locks
                print("ChromaDB client reset successfully.")
                logger.info("ChromaDB client reset.")
                time.sleep(5.5)
            except Exception as e:
                print(f"Error resetting ChromaDB client: {e}")
                logger.error(f"Error resetting ChromaDB client: {e}", exc_info=True)

        _embedding_model = None  # Allow garbage collection
        _chroma_client = None
        _chroma_collection = None
        logger.info("Retrieval service resources released.")


class VectorSearchService:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        chroma_collection: chromadb.Collection,
        top_k: int,
        distance_threshold: float = 1.0,  # Adjust later this value based on experimentation
    ):
        self.embedding_model = embedding_model
        self.chroma_collection = chroma_collection
        self.top_k = top_k
        self.distance_threshold = distance_threshold  # Store the threshold

    def _embed_query(self, query: str) -> List[float]:
        """Generates embedding for the given query text."""
        # Note: encode() returns a numpy array, convert to list for ChromaDB
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            # Some models might output multi-dimensional arrays, ensure it's 1D
            if embedding.ndim > 1:
                embedding = (
                    embedding.flatten()
                )  # Or handle appropriately based on model output
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate query embedding.",
            ) from e

    async def search(self, query: str) -> List[str]:
        """
        Embeds the query and performs similarity search in ChromaDB.

        Returns:
            A list of relevant document chunk texts.
        """
        logger.info(f"Search service using collection: '{self.chroma_collection.name}'")
        logger.info(f"Embedding query: '{query[:50]}...'")
        query_embedding = self._embed_query(query)

        logger.info(
            f"Querying ChromaDB collection '{self.chroma_collection.name}' for top {self.top_k} results..."
        )
        try:
            results = self.chroma_collection.query(
                query_embeddings=[
                    query_embedding
                ],  # Chroma expects a list of embeddings
                n_results=self.top_k,
                include=[
                    "documents",
                    "distances",
                ],  # We only need the document text content
            )
            logger.debug(f"Raw ChromaDB query results: {results}")

            # Filter results based on distance
            filtered_chunks = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                distances = results["distances"][0]

                for doc, dist in zip(documents, distances):
                    if dist <= self.distance_threshold:
                        filtered_chunks.append(doc)
                    else:
                        logger.debug(
                            f"Filtered out chunk due to distance {dist} > {self.distance_threshold}"
                        )

            logger.info(
                f"Returning {len(filtered_chunks)} chunks after distance filtering."
            )
            return filtered_chunks

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to query vector database: {e}",
            ) from e

    async def add_documents(self, documents: Dict[str, str]) -> int:
        """
        Embeds and adds new documents to the ChromaDB collection.
        Returns the number of documents added.
        """
        if not documents:
            logger.warning("Add documents called with an empty dictionary.")
            return 0

        doc_ids = list(documents.keys())
        doc_texts = list(documents.values())
        logger.info(
            f"Adding {len(doc_ids)} documents to collection '{self.chroma_collection.name}'..."
        )

        try:
            # Generate embeddings
            logger.debug("Generating embeddings for new documents...")
            embeddings = self.embedding_model.encode(doc_texts, convert_to_tensor=False)
            logger.debug(f"Generated {len(embeddings)} embeddings.")

            self.chroma_collection.add(
                ids=doc_ids, documents=doc_texts, embeddings=embeddings
            )
            logger.info(f"Successfully added/updated {len(doc_ids)} documents.")
            return len(doc_ids)

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to add documents to the collection: {e}",
            )
