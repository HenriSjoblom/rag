import asyncio
import logging
from typing import Any, Dict, List

from app.config import Settings
from app.services.chroma_manager import ChromaClientManager
from app.services.embedding_manager import EmbeddingModelManager
from app.services.vector_store_manager import VectorStoreManager
from chromadb.errors import ChromaError
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class VectorSearchService:
    """Handles vector search operations for retrieval only."""

    def __init__(
        self,
        settings: Settings,
        chroma_manager: ChromaClientManager,
        embedding_manager: EmbeddingModelManager,
        vector_store_manager: VectorStoreManager,
    ):
        self.settings = settings
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
        self.vector_store_manager = vector_store_manager
        logger.info("VectorSearchService initialized for retrieval operations.")

    def _embed_query(self, query: str) -> List[float]:
        """Generates embedding for the given query text."""
        if not query or not query.strip():
            logger.error("Empty query provided for embedding")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty.",
            )

        try:
            embedding_model = self.embedding_manager.get_model()
            embedding = embedding_model.encode(query.strip(), convert_to_numpy=True)

            if embedding is None:
                raise ValueError("Embedding model returned None")

            if embedding.ndim > 1:
                embedding = embedding.flatten()

            embedding_list = embedding.tolist()

            if not embedding_list or len(embedding_list) == 0:
                raise ValueError("Embedding model returned empty result")

            return embedding_list

        except MemoryError as e:
            logger.error(f"Out of memory during query embedding: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail="Insufficient memory to process query embedding.",
            ) from e
        except ValueError as e:
            logger.error(f"Invalid input for query embedding: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid query format: {str(e)}",
            ) from e
        except Exception as e:
            logger.error(f"Failed to embed query: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate query embedding.",
            ) from e

    async def _get_fresh_collection(self):
        """Get a fresh collection instance."""
        try:
            return await asyncio.to_thread(self.vector_store_manager.get_collection)
        except ConnectionError as e:
            logger.error(f"Connection error getting collection: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector database is currently unavailable.",
            ) from e
        except RuntimeError as e:
            logger.error(f"Runtime error getting collection: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Vector database collection not available: {str(e)}",
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error getting collection: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to access vector database collection.",
            ) from e

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []

        try:
            collection = await self._get_fresh_collection()
            logger.info(
                f"Search service using collection: '{collection.name}' (ID: {collection.id}) for query: '{query[:50]}...'"
            )
            query_embedding = self._embed_query(query)

            if self.settings.TOP_K_RESULTS <= 0:
                raise ValueError("TOP_K_RESULTS must be greater than 0")

            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=self.settings.TOP_K_RESULTS,
                include=["documents", "metadatas", "distances"],
            )

            if not results:
                logger.info("No results returned from ChromaDB query")
                return []

            logger.debug(f"Raw ChromaDB query results: {results}")

            processed_chunks = []
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
                    try:
                        doc_id = ids_list[i]
                        doc_text = documents_list[i]
                        metadata = metadatas_list[i]
                        distance = distances_list[i]

                        if doc_id and doc_text:  # Only include valid results
                            processed_chunks.append(
                                {
                                    "id": doc_id,
                                    "text": doc_text,
                                    "metadata": metadata if metadata else {},
                                    "distance": distance
                                    if distance is not None
                                    else float("inf"),
                                }
                            )
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Skipping malformed result at index {i}: {e}")
                        continue

            logger.info(
                f"Returning {len(processed_chunks)} chunks for query '{query[:50]}...'."
            )
            return processed_chunks

        except HTTPException:
            raise
        except ChromaError as e:
            logger.error(f"ChromaDB query error: {e}", exc_info=True)
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Vector database collection '{self.settings.CHROMA_COLLECTION_NAME}' not found.",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error during search operation.",
            ) from e
        except ValueError as e:
            logger.error(f"Configuration error during search: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search configuration error: {str(e)}",
            ) from e
        except Exception as e:
            logger.error(
                f"ChromaDB query failed for collection '{self.settings.CHROMA_COLLECTION_NAME}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to query vector database: {str(e)}",
            ) from e
