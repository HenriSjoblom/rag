import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import get_vector_search_service
from app.models import RetrievalRequest, RetrievalResponse
from app.services.vector_search import VectorSearchService

router = APIRouter(tags=["retrieval"])
logger = logging.getLogger(__name__)


@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve relevant document chunks",
    description="Receives a query, embeds it, and retrieves the most relevant text chunks from the vector database.",
)
async def retrieve_chunks(
    request: RetrievalRequest,
    search_service: VectorSearchService = Depends(get_vector_search_service),
):
    """
    Handles incoming requests to retrieve relevant document chunks.
    """
    logger.info(f"Processing retrieval request for query: '{request.query[:50]}...'")
    try:
        # Validate query length
        if len(request.query) > 10000:  # Reasonable limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query too long. Maximum length is 10,000 characters.",
            )

        # Perform the search using the injected service
        retrieved_items = await search_service.search(query=request.query)

        # Extract only the text from each retrieved item
        text_chunks = [
            item["text"] for item in retrieved_items if item.get("text") is not None
        ]

        # Return the results in the expected format
        return RetrievalResponse(
            chunks=text_chunks,
            collection_name=search_service.settings.CHROMA_COLLECTION_NAME,
            query=request.query,
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions raised by the service layer
        logger.warning(f"HTTP error in retrieval: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in retrieval endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during retrieval.",
        )
