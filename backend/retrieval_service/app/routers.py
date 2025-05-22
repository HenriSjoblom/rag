import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import get_vector_search_service

from app.models import (
    RetrievalRequest,
    RetrievalResponse,
)
from app.services.vector_search import VectorSearchService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve relevant document chunks",
    description="Receives a query, embeds it, and retrieves the most relevant text chunks from the vector database.",
    tags=["retrieval"],
)
async def retrieve_chunks(
    request: RetrievalRequest,
    search_service: VectorSearchService = Depends(
        get_vector_search_service
    ),  # Inject service
):
    """
    Handles incoming requests to retrieve relevant document chunks.
    """
    logger.info(
        f"Router using collection: '{search_service.collection_name}' for retrieval of query: '{request.query}'"
    )
    try:
        # Perform the search using the injected service
        retrieved_items = await search_service.search(query=request.query)

        # Extract only the text from each retrieved item
        text_chunks = [
            item["text"] for item in retrieved_items if item.get("text") is not None
        ]

        # Return the results in the expected format
        return RetrievalResponse(chunks=text_chunks)
    except HTTPException as e:
        # Re-raise HTTPExceptions raised by the service layer
        raise e
    except Exception as e:
        logger.error(
            f"Unexpected error in retrieval endpoint: {e}", exc_info=True
        ) 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during retrieval.",
        )