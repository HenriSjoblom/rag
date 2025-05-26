import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import get_generation_service
from app.models import GenerateRequest, GenerateResponse
from app.services.generation import GenerationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["generation"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate an answer using LLM based on query and context",
    description="Receives a user query and retrieved context chunks, formats a prompt, calls the configured LLM, and returns the generated answer.",
)
async def generate_answer(
    request: GenerateRequest,
    generation_service: GenerationService = Depends(get_generation_service),
):
    """
    Generate an answer using the configured LLM based on provided query and context.

    This endpoint processes the user's query along with relevant context chunks,
    formats them into a suitable prompt, and returns the LLM's generated response.

    Args:
        request: GenerateRequest containing query and context chunks
        generation_service: Injected GenerationService instance

    Returns:
        GenerateResponse: Contains the generated answer

    Raises:
        HTTPException: For various error conditions (400, 500, 503)
    """
    try:
        # Log incoming request details
        query_preview = (
            request.query[:50] + "..." if len(request.query) > 50 else request.query
        )
        logger.info(f"Generation request received for query: '{query_preview}'")
        logger.debug(f"Request contains {len(request.context_chunks)} context chunks")

        # Basic request validation
        if not request.query.strip():
            logger.warning("Empty query received in generation request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty",
            )

        # Process the generation request
        answer = await generation_service.generate_answer(request)

        # Log successful response
        logger.info(f"Successfully generated answer (length: {len(answer)} chars)")
        if logger.isEnabledFor(logging.DEBUG):
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            logger.debug(f"Generated answer preview: '{answer_preview}'")

        return GenerateResponse(answer=answer)

    except HTTPException as e:
        # Re-raise service-level HTTP exceptions (e.g., 503 from LLM failures)
        logger.warning(f"Service-level HTTP error in generation: {e.detail}")
        raise e
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error in generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request data: {str(e)}",
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error during generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating the response.",
        )
