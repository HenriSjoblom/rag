from fastapi import APIRouter, Depends, HTTPException, status

from app.models import ChatRequest, ChatResponse
from app.services.chat_processor import ChatProcessorService
from app.deps import get_chat_processor_service

router = APIRouter()

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process a user chat message",
    description="Receives a user message, orchestrates RAG pipeline and returns the response.",
    tags=["chat"]
)
async def handle_chat_message(
    request: ChatRequest,
    chat_service: ChatProcessorService = Depends(get_chat_processor_service) # Inject the service
):
    """
    Handles incoming chat messages.
    """
    try:
        print(f"Received chat request: {request}")
        response = await chat_service.process_message(request)
        return ChatResponse(response=response)
    except HTTPException as e:
        # If the service layer raised an HTTPException, re-raise it
        raise e
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Unexpected error in chat endpoint: {e}") # Log the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your message.",
        )