from fastapi import APIRouter, Depends, HTTPException, status

from app.models import GenerateRequest, GenerateResponse
from app.services.generation import GenerationService
from app.deps import get_generation_service

router = APIRouter()

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate an answer using LLM based on query and context",
    description="Receives a user query and retrieved context chunks, formats a prompt, calls the configured LLM, and returns the generated answer.",
    tags=["Generation"]
)
async def handle_generation_request(
    request: GenerateRequest,
    gen_service: GenerationService = Depends(get_generation_service) # Inject the cached service
):
    """
    Handles incoming requests to generate a response.
    """
    try:
        answer = await gen_service.generate_answer(request)
        return GenerateResponse(answer=answer)
    except HTTPException as e:
        # Re-raise HTTPExceptions raised by the service layer (e.g., LLM unavailable)
        print(f"HTTPException in generation endpoint: {e.detail}")
        raise e
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Unexpected error in generation endpoint: {e}") # Log the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating the response.",
        )