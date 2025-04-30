from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.models.retrieval import RetrievalRequest, RetrievalResponse
from app.services.vector_search import VectorSearchService
from app.api.deps import get_vector_search_service

router = APIRouter()

@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve relevant document chunks",
    description="Receives a query, embeds it, and retrieves the most relevant text chunks from the vector database.",
    tags=["Retrieval"]
)
async def retrieve_chunks(
    request: RetrievalRequest,
    search_service: VectorSearchService = Depends(get_vector_search_service) # Inject service
):
    """
    Handles incoming requests to retrieve relevant document chunks.
    """
    try:
        # Perform the search using the injected service
        retrieved_chunks = await search_service.search(query=request.query)
        # Return the results in the expected format
        return RetrievalResponse(chunks=retrieved_chunks)
    except HTTPException as e:
        # Re-raise HTTPExceptions raised by the service layer
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error in retrieval endpoint: {e}") # Log the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during retrieval.",
        )