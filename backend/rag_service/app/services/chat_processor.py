import logging
from typing import Any, Dict, List

import httpx
from app.models import (
    GenerationRequest,
    GenerationResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from fastapi import HTTPException, status

from .http_client import make_request

logger = logging.getLogger(__name__)


class ChatProcessorService:
    def __init__(
        self,
        retrieval_service_url: str,
        generation_service_url: str,
        http_client: httpx.AsyncClient,
    ):
        self.retrieval_service_url = retrieval_service_url
        self.generation_service_url = generation_service_url
        self.http_client = http_client

    async def _call_retrieval_service(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:  # Return type is List of Dictionaries
        """Calls the retrieval microservice."""
        retrieval_url = f"{str(self.retrieval_service_url).rstrip('/')}/api/v1/retrieve"
        payload = RetrievalRequest(query=query, top_k=top_k)

        logger.debug(
            f"Calling retrieval service at {retrieval_url} with payload: {payload.model_dump_json(indent=2)}"
        )

        try:
            response_data = await make_request(
                client=self.http_client,
                method="POST",
                url=retrieval_url,
                json_data=payload.model_dump(),
            )

            try:
                validated_response = RetrievalResponse.model_validate(response_data)

                processed_chunks = [
                    {"text": chunk_text} for chunk_text in validated_response.chunks
                ]

                logger.info(
                    f"Received and processed {len(processed_chunks)} chunks from retrieval service."
                )
                return processed_chunks

            except (
                Exception
            ) as val_err:  # Handles Pydantic validation errors or other issues
                logger.error(
                    f"Failed to validate or process response from retrieval service: {val_err}. Response data: {response_data}"
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Retrieval service returned data that failed validation or processing.",
                )
        except HTTPException as e:
            logger.error(
                f"HTTPException from retrieval service: {e.status_code} - {e.detail}"
            )
            raise  # Re-raise the exception to be handled by the caller
        except Exception as e:
            logger.exception(
                f"Unexpected error calling retrieval service at {retrieval_url}: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Retrieval service is unavailable or encountered an error: {str(e)}",
            )

    async def _call_generation_service(
        self,
        user_query: str,
        context_chunks: List[Dict[str, Any]],  # This is List[{"text": "..."}]
    ) -> str:
        """Calls the generation microservice."""
        # Use the passed-in URL
        generation_url = (
            f"{str(self.generation_service_url).rstrip('/')}/api/v1/generate"
        )

        prepared_context_chunks = [
            chunk.get("text", "")
            for chunk in context_chunks
            if chunk.get("text") is not None  # Ensure text key exists and is not None
        ]

        payload = GenerationRequest(
            query=user_query, context_chunks=prepared_context_chunks
        )
        logger.debug(
            f"Calling generation service at {generation_url} with payload: {payload.model_dump_json(indent=2)}"
        )

        try:
            response_data = await make_request(
                client=self.http_client,
                method="POST",
                url=generation_url,
                json_data=payload.model_dump(),
            )
            # Validate with GenerationResponse model
            validated_response = GenerationResponse.model_validate(response_data)
            logger.info("Received response from generation service.")
            return validated_response.answer
        except HTTPException as e:
            logger.error(
                f"HTTPException from generation service: {e.status_code} - {e.detail}"
            )
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error calling generation service at {generation_url}: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Generation service is unavailable or encountered an error: {str(e)}",
            )

    async def process(self, user_id: str, query: str) -> str:
        """
        Orchestrates the RAG pipeline:
        1. Calls retrieval service to get context.
        2. Calls generation service with query and context to get an answer.
        """
        logger.info(
            f"Processing chat for user_id: {user_id}, query: '{query[:100]}...'"
        )

        # Call Retrieval Service
        try:
            retrieved_chunks = await self._call_retrieval_service(query=query)
            if not retrieved_chunks:
                logger.warning(
                    "No context chunks retrieved. Proceeding without context."
                )
                # Depending on requirements, you might return a specific message or proceed
                # return "I couldn't find any specific information related to your query. Can you try rephrasing?"
        except HTTPException as e:
            # Specific handling for retrieval failure if needed, or re-raise
            logger.error(f"Failed to retrieve context for query '{query}': {e.detail}")
            raise HTTPException(
                status_code=e.status_code, detail=f"Error from retrieval: {e.detail}"
            )

        # Call Generation Service
        try:
            ai_response = await self._call_generation_service(
                user_query=query, context_chunks=retrieved_chunks
            )
            logger.info(f"Generated AI response for user_id: {user_id}")
            return ai_response
        except HTTPException as e:
            logger.error(f"Failed to generate response for query '{query}': {e.detail}")
            raise HTTPException(
                status_code=e.status_code, detail=f"Error from generation: {e.detail}"
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error during generation for query '{query}': {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while generating a response.",
            )
