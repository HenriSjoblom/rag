import httpx
from typing import List
from fastapi import Depends, HTTPException, status

from app.config import Settings
from app.models import (
    RetrievalRequest,
    RetrievalResponse,
    GenerationRequest,
    GenerationResponse,
    ChatRequest,
)
from .http_client import make_request, get_http_client

class ChatProcessorService:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient
    ):
        self.settings = settings
        self.http_client = http_client

    async def _call_retrieval_service(self, query: str) -> List[str]:
        """Calls the retrieval microservice."""
        retrieval_url = f"{str(self.settings.RETRIEVAL_SERVICE_URL).rstrip('/')}/api/v1/retrieve"
        payload = RetrievalRequest(query=query)
        print(f"Calling retrieval service at {retrieval_url} with payload: {payload.model_dump()}")
        print(f"Payload model dump: {payload.model_dump_json(indent=2)}")
        try:
            response_data = await make_request(
                client=self.http_client,
                method="POST",
                url=retrieval_url,
                json_data=payload.model_dump()
            )
            # Validate response structure (basic check)
            print(f"Response data from retrieval service: {response_data}")
            print(f"make request id: {id(make_request)}")
            retrieval_response = RetrievalResponse(**response_data)
            return retrieval_response.chunks
        except HTTPException as e:
             # Re-raise or handle specific errors from retrieval
            print(f"Error calling retrieval service: {e.detail}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Retrieval service failed: {e.detail}"
            ) from e
        except Exception as e: # Catch unexpected validation or other errors
            print(f"Unexpected error during retrieval call: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while retrieving context."
            ) from e


    async def _call_generation_service(self, query: str, context_chunks: List[str]) -> str:
        """Calls the generation microservice."""
        generation_url = f"{str(self.settings.GENERATION_SERVICE_URL).rstrip('/')}/api/v1/generate" # Assuming /generate endpoint
        payload = GenerationRequest(query=query, context_chunks=context_chunks)
        try:
            response_data = await make_request(
                client=self.http_client,
                method="POST",
                url=generation_url,
                json_data=payload.model_dump()
            )
             # Validate response structure (basic check)
            generation_response = GenerationResponse(**response_data)
            return generation_response.answer
        except HTTPException as e:
             # Re-raise or handle specific errors from generation
            print(f"Error calling generation service: {e.detail}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Generation service failed: {e.detail}"
            ) from e
        except Exception as e: # Catch unexpected validation or other errors
            print(f"Unexpected error during generation call: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while generating the response."
            ) from e

    async def process_message(self, request: ChatRequest) -> str:
        """
        Orchestrates the RAG process for a user message.

        1. Calls retrieval service.
        2. Calls generation service with query and retrieved context.
        3. Returns the final answer.
        """
        print(f"Processing message from user: {request.user_id}")

        # Retrieve relevant context chunks
        print(f"Calling retrieval service for query: '{request.message[:50]}...'")
        retrieved_chunks = await self._call_retrieval_service(query=request.message)
        print(f"Retrieved {len(retrieved_chunks)} chunks.")
        if not retrieved_chunks:
             # Decide how to handle no context: fallback message or try generating without?
             # For now, let's provide a fallback.
             print("No relevant context found.")
             # return "I couldn't find specific information about that in my knowledge base. Can you please rephrase or ask something else?"
             # Or proceed to generation, letting the LLM handle lack of context (might hallucinate more)

        # Generate the final answer using the LLM
        print("Calling generation service...")
        final_answer = await self._call_generation_service(
            query=request.message,
            context_chunks=retrieved_chunks
        )
        print("Generation complete.")

        return final_answer