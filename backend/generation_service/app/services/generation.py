import os
from typing import List, Dict, Any, Optional

from fastapi import HTTPException, status
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.models import GenerateRequest

# --- Prompt Template ---

# A more robust prompt for RAG
RAG_PROMPT_TEMPLATE = """
SYSTEM: You are a helpful and precise customer support assistant. Your goal is to answer the user's query based *only* on the provided context.
- If the context contains the information needed to answer the query, provide a clear and concise answer citing the relevant information from the context.
- If the context does not contain information relevant to the query, politely state that you don't have enough information based on the provided documents. Do not make up information or use external knowledge.
- If the query is a greeting or conversational filler, respond politely as a support assistant.

CONTEXT:
{context}

USER QUERY:
{query}

ASSISTANT RESPONSE:
"""

class GenerationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chat_model = self._initialize_llm()
        self.prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.output_parser = StrOutputParser()

        # Define the LangChain Expression Language (LCEL) chain
        self.rag_chain: Runnable = (
            self.prompt_template
            | self.chat_model
            | self.output_parser
        )
        print(f"Initialized LLM Service with provider '{settings.LLM_PROVIDER}' and model '{settings.LLM_MODEL_NAME}'")


    def _initialize_llm(self):
        """Initializes the LangChain ChatModel based on settings."""
        provider = self.settings.LLM_PROVIDER
        model_name = self.settings.LLM_MODEL_NAME
        temperature = self.settings.LLM_TEMPERATURE
        max_tokens = self.settings.LLM_MAX_TOKENS

        if provider == "openai":
            if not self.settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not configured.")

            api_key = self.settings.OPENAI_API_KEY.get_secret_value()

            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key # Pass key directly if preferred
            )
        # Add more providers later
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _format_context(self, context_chunks: List[str]) -> str:
        """Formats the list of context chunks into a single string for the prompt."""
        if not context_chunks:
            return "No context provided."
        # Combine chunks with separators
        return "\n---\n".join(context_chunks)

    async def generate_answer(self, request: GenerateRequest) -> str:
        """Formats the prompt, invokes the LLM chain, and returns the answer."""
        print(f"Generating answer for query: '{request.query[:50]}...'")
        print(f"Received {len(request.context_chunks)} context chunks.")

        formatted_context = self._format_context(request.context_chunks)

        # Prepare input for the LCEL chain
        chain_input = {
            "context": formatted_context,
            "query": request.query
        }

        try:
            # Use ainvoke for asynchronous execution with LCEL
            print("Invoking RAG chain...")
            result = await self.rag_chain.ainvoke(chain_input)
            print(f"LLM Response received: '{result[:100]}...'")
            return result

        except Exception as e:
            # Catch potential errors from the LLM API (rate limits, auth issues etc.)
            # LangChain might wrap these, or they might be raw httpx/openai errors
            print(f"Error invoking LLM chain: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to get response from LLM: {e}",
            ) from e