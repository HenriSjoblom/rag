import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import get_chat_processor_service
from app.models import ChatRequest, ChatResponse, ServiceErrorResponse
from app.services.chat_processor import ChatProcessorService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process a user's chat message",
    description="Receives a user message, orchestrates retrieval and generation, and returns an AI response.",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceErrorResponse,
            "description": "Downstream service is unavailable",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ServiceErrorResponse,
            "description": "Internal server error",
        },
    },
)
async def process_chat_message(
    chat_request: ChatRequest,
    chat_processor: ChatProcessorService = Depends(get_chat_processor_service),
):
    try:
        logger.info(f"Received chat request, message: '{chat_request.message[:50]}...'")
        ai_response_text = await chat_processor.process(query=chat_request.message)
        logger.info("Successfully processed chat request")
        return ChatResponse(
            query=chat_request.message,
            response=ai_response_text,
        )
    except HTTPException as http_exc:  # Re-raise known HTTP exceptions
        logger.error(f"HTTPException during chat processing: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(
            f"Unexpected error processing chat: {e}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
