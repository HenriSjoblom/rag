import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from fastapi import HTTPException, status
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global instances managed by lifespan
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None
_chroma_collection: Optional[chromadb.Collection] = (
    None  # This global one can become stale
)


async def get_embedding_model() -> SentenceTransformer:
    """Dependency to get the loaded embedding model."""
    if _embedding_model is None:
        # This should ideally not happen if lifespan management is correct
        logger.error("Embedding model accessed before initialization.")
        raise RuntimeError("Embedding model not initialized.")
    return _embedding_model


async def get_chroma_client() -> (
    chromadb.ClientAPI
):  # Changed from get_chroma_collection
    """Dependency to get the ChromaDB client API object."""
    if _chroma_client is None:
        # This should ideally not happen
        logger.error("ChromaDB client accessed before initialization.")
        raise RuntimeError("ChromaDB client not initialized.")
    return _chroma_client


@asynccontextmanager
async def lifespan_retrieval_service(
    app,  # FastAPI app instance, often implicitly passed or not needed by user code
    model_name: str,
    chroma_mode: str,
    collection_name: str,
    chroma_path: Optional[str] = None,
    chroma_host: Optional[str] = None,
    chroma_port: Optional[int] = None,
):
    """Manages the lifespan of embedding model and ChromaDB client."""
    global _embedding_model, _chroma_client, _chroma_collection  # _chroma_collection is for initial setup/check

    # --- Idempotency Check for testing/hot-reloading ---
    if (
        _embedding_model is not None and _chroma_client is not None
    ):  # Check client instead of collection
        logger.info(
            "Lifespan: Resources appear to be already initialized. Skipping setup."
        )
        try:
            yield
        finally:
            # In this scenario, we don't own the teardown if setup was skipped.
            # However, for robust testing, ensure no old resources are accidentally reset.
            pass
        return

    logger.info("Initializing retrieval service resources...")
    logger.info(f"Loading embedding model: {model_name}...")
    try:
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(
            f"Failed to load embedding model '{model_name}': {e}", exc_info=True
        )
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

    logger.info(f"Connecting to ChromaDB in '{chroma_mode}' mode...")
    if chroma_mode == "local":
        if not chroma_path:
            logger.error("chroma_path is required for local ChromaDB mode.")
            raise ValueError("chroma_path is required for local mode.")
        logger.info(f"ChromaDB local path: {chroma_path}")
        _chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(allow_reset=True),  # allow_reset for dev/testing
        )
    elif chroma_mode == "docker":
        if not chroma_host or not chroma_port:
            logger.error(
                "chroma_host and chroma_port are required for Docker ChromaDB mode."
            )
            raise ValueError(
                "chroma_host and chroma_port are required for docker mode."
            )
        logger.info(f"ChromaDB Docker host: {chroma_host}, port: {chroma_port}")
        _chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        logger.error(
            f"Invalid CHROMA_MODE: {chroma_mode}. Must be 'local' or 'docker'."
        )
        raise ValueError(
            f"Invalid CHROMA_MODE: {chroma_mode}. Must be 'local' or 'docker'."
        )
    logger.info("ChromaDB client initialized.")

    try:
        # Ensure collection is created with the same embedding function characteristics
        # as the model_name we are using for explicit embeddings.
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        logger.info(
            f"Attempting to get or create ChromaDB collection '{collection_name}' with EF for model '{model_name}'..."
        )
        # Store the initial collection reference for potential direct use or checks if needed,
        # but VectorSearchService will re-fetch by name.
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_ef,  # Crucial for consistency
        )
        logger.info(
            f"ChromaDB collection '{_chroma_collection.name}' (ID: {_chroma_collection.id}) is available."
        )
    except Exception as e:
        logger.error(
            f"Failed to initialize ChromaDB collection '{collection_name}': {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Failed to initialize ChromaDB collection: {e}") from e

    try:
        yield  # Application runs
    finally:
        logger.info("Shutting down retrieval service resources...")
        if (
            _chroma_client
            and hasattr(_chroma_client, "reset")
            and callable(getattr(_chroma_client, "reset"))
        ):
            try:
                logger.info("Resetting ChromaDB client...")
                _chroma_client.reset()
                logger.info("ChromaDB client reset successfully.")
                # Add a small delay if needed, observed in user code
                # time.sleep(1) # Example: time.sleep(config.CHROMA_RESET_DELAY)
            except Exception as e:
                logger.error(f"Error resetting ChromaDB client: {e}", exc_info=True)

        _embedding_model = None
        _chroma_client = None
        _chroma_collection = None
        logger.info("Retrieval service resources released.")


class VectorSearchService:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        chroma_client: chromadb.ClientAPI,  # Changed
        collection_name: str,  # Added
        top_k: int,
        distance_threshold: float = 1.0,
    ):
        self.embedding_model = embedding_model
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.top_k = top_k
        self.distance_threshold = distance_threshold

    async def _get_fresh_collection(self) -> chromadb.Collection:
        """Fetches a fresh collection object from ChromaDB by name."""
        try:
            # When getting an existing collection, Chroma uses the EF it was created with.
            # So, no need to pass embedding_function to get_collection here if it was set at creation.
            collection = await asyncio.to_thread(
                self.chroma_client.get_collection, name=self.collection_name
            )
            logger.debug(
                f"Successfully fetched fresh collection '{collection.name}' (ID: {collection.id})."
            )
            return collection
        except Exception as e:
            # This includes cases where the collection doesn't exist.
            logger.error(
                f"Failed to get ChromaDB collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Vector database collection '{self.collection_name}' not found or inaccessible: {str(e)}",
            )

    def _embed_query(self, query: str) -> List[float]:
        """Generates embedding for the given query text."""
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate query embedding.",
            ) from e

    async def search(self, query: str) -> List[Dict[str, Any]]:
        collection = await self._get_fresh_collection()
        logger.info(
            f"Search service using collection: '{collection.name}' (ID: {collection.id}) for query: '{query[:50]}...'"
        )
        query_embedding = self._embed_query(query)

        try:
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],  # Removed "ids" from here
            )
            logger.debug(f"Raw ChromaDB query results: {results}")

            processed_chunks = []
            # The results dictionary will still contain an 'ids' key at the top level
            if results and results.get("ids") and results["ids"][0]:
                ids_list = results["ids"][0]
                # Ensure other lists are handled safely if not included or empty
                documents_list = (
                    results.get("documents")[0]
                    if results.get("documents") and results["documents"]
                    else [None] * len(ids_list)
                )
                metadatas_list = (
                    results.get("metadatas")[0]
                    if results.get("metadatas") and results["metadatas"]
                    else [{}] * len(ids_list)
                )
                distances_list = (
                    results.get("distances")[0]
                    if results.get("distances") and results["distances"]
                    else [float("inf")] * len(ids_list)
                )

                for i in range(len(ids_list)):
                    doc_id = ids_list[i]
                    doc_text = documents_list[i]
                    metadata = metadatas_list[i]
                    distance = distances_list[i]

                    if distance is not None and distance <= self.distance_threshold:
                        processed_chunks.append(
                            {
                                "id": doc_id,  # We still use the ID here
                                "text": doc_text,
                                "metadata": metadata if metadata else {},
                                "distance": distance,
                            }
                        )
                    else:
                        logger.debug(
                            f"Filtered out chunk ID {doc_id} (text: {str(doc_text)[:30]}...) due to distance {distance} > {self.distance_threshold}"
                        )

            logger.info(
                f"Returning {len(processed_chunks)} chunks after distance filtering for query '{query[:50]}...'."
            )
            return processed_chunks
        except Exception as e:
            logger.error(
                f"ChromaDB query failed for collection '{collection.name}': {e}",
                exc_info=True,
            )
            # Check if the error is about the collection not existing, which _get_fresh_collection should catch
            # but re-checking here can provide a more specific error context if it somehow passes _get_fresh_collection
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Vector database collection '{self.collection_name}' could not be found during query operation: {str(e)}",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to query vector database: {str(e)}",
            ) from e

    async def add_documents(self, documents: Dict[str, str]) -> int:
        collection = await self._get_fresh_collection()
        if not documents:
            logger.warning("Add documents called with an empty dictionary.")
            return 0

        doc_ids = list(documents.keys())
        doc_texts = list(documents.values())
        logger.info(
            f"Adding {len(doc_ids)} documents to collection '{collection.name}' (ID: {collection.id})..."
        )

        try:
            logger.debug("Generating embeddings for new documents...")
            # SentenceTransformer.encode can be CPU-bound, run in thread
            embeddings_np = await asyncio.to_thread(
                self.embedding_model.encode, doc_texts
            )
            # Ensure embeddings are in the format List[List[float]]
            embeddings_list = [emb.tolist() for emb in embeddings_np]
            logger.debug(f"Generated {len(embeddings_list)} embeddings.")

            await asyncio.to_thread(
                collection.add,
                ids=doc_ids,
                documents=doc_texts,
                embeddings=embeddings_list,
            )
            logger.info(
                f"Successfully added/updated {len(doc_ids)} documents to collection '{collection.name}'."
            )
            return len(doc_ids)
        except Exception as e:
            logger.error(
                f"Error adding documents to ChromaDB collection '{collection.name}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to add documents to collection '{self.collection_name}': {e}",
            ) from e
