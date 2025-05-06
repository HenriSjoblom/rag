from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import logging # Import logging

# Import the new models
from app.models import RetrievalRequest, RetrievalResponse, AddDataRequest, AddDataResponse
from app.services.vector_search import VectorSearchService
from app.deps import get_vector_search_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve relevant document chunks",
    description="Receives a query, embeds it, and retrieves the most relevant text chunks from the vector database.",
    tags=["retrieval"]
)
async def retrieve_chunks(
    request: RetrievalRequest,
    search_service: VectorSearchService = Depends(get_vector_search_service) # Inject service
):
    """
    Handles incoming requests to retrieve relevant document chunks.
    """
    logger.info(f"Router using collection: '{search_service.chroma_collection.name}' for retrieval") # Use logger
    try:
        # Perform the search using the injected service
        retrieved_chunks = await search_service.search(query=request.query)
        # Return the results in the expected format
        return RetrievalResponse(chunks=retrieved_chunks)
    except HTTPException as e:
        # Re-raise HTTPExceptions raised by the service layer
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in retrieval endpoint: {e}", exc_info=True) # Use logger
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during retrieval.",
        )

@router.post(
    "/add",
    response_model=AddDataResponse,
    summary="Add new documents to the collection",
    description="Receives a dictionary of document IDs and texts, embeds them, and adds them to the vector database.",
    tags=["data management"]
)
async def add_new_documents(
    request: AddDataRequest,
    search_service: VectorSearchService = Depends(get_vector_search_service) # Inject service
):
    """
    Handles incoming requests to add new documents.
    """
    logger.info(f"Router using collection: '{search_service.chroma_collection.name}' for adding data")
    if not request.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents provided in the request.",
        )
    try:
        added_count = await search_service.add_documents(documents=request.documents)
        return AddDataResponse(
            added_count=added_count,
            collection_name=search_service.chroma_collection.name
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions (like the 500 error from the service)
        raise e
    except Exception as e:
        # Catch any other unexpected errors during the request handling
        logger.error(f"Unexpected error in add endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the add request.",
        )