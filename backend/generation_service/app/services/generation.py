import logging
from typing import List

from fastapi import HTTPException, status
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.models import GenerateRequest

logger = logging.getLogger(__name__)

# RAG prompt template for user manual assistance
RAG_PROMPT_TEMPLATE = """
You are a helpful technical documentation assistant specializing in user manuals and product guides. Your primary goal is to help users understand how to use products and troubleshoot issues based on the provided documentation.

Instructions:
- Answer questions using ONLY the information provided in the context below
- Provide clear, step-by-step instructions when explaining procedures
- Include relevant warnings, cautions, or safety notes mentioned in the documentation
- If the context contains multiple relevant sections, synthesize the information coherently
- When referencing specific features, buttons, or settings, use the exact terminology from the manual
- If the context doesn't contain sufficient information to answer the question, clearly state this limitation
- For troubleshooting questions, provide systematic diagnostic steps if available in the context
- Always prioritize user safety and proper usage guidelines

CONTEXT FROM USER MANUAL:
{context}

USER QUESTION:
{query}

ASSISTANT RESPONSE:
Based on the user manual information provided:

"""


class GenerationService:
    """
    Service for generating responses using LLM with RAG context.

    This service initializes a LangChain pipeline that combines:
    - A prompt template for RAG scenarios
    - A configured LLM (currently OpenAI)
    - Output parsing for clean responses
    """

    def __init__(self, settings: Settings):
        """
        Initialize the generation service with LLM and prompt configuration.

        Args:
            settings: Application settings containing LLM configuration

        Raises:
            RuntimeError: If service initialization fails
        """
        try:
            self.settings = settings
            logger.info(
                f"Initializing GenerationService with provider: {settings.LLM_PROVIDER}"
            )

            # Initialize LLM components in order
            self.chat_model = self._initialize_llm()
            self.prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            self.output_parser = StrOutputParser()

            # Create the LangChain Expression Language (LCEL) chain
            self.rag_chain: Runnable = (
                self.prompt_template | self.chat_model | self.output_parser
            )

            logger.info(
                f"GenerationService initialized successfully with model '{settings.LLM_MODEL_NAME}'"
            )

        except Exception as e:
            logger.error(f"Failed to initialize GenerationService: {e}", exc_info=True)
            raise RuntimeError(f"GenerationService initialization failed: {e}") from e

    def _initialize_llm(self):
        """
        Initialize the LangChain ChatModel based on provider settings.

        Returns:
            ChatModel: Configured LLM instance

        Raises:
            ValueError: If provider is unsupported or configuration is invalid
        """
        provider = self.settings.LLM_PROVIDER
        model_name = self.settings.LLM_MODEL_NAME
        temperature = self.settings.LLM_TEMPERATURE
        max_tokens = self.settings.LLM_MAX_TOKENS

        logger.debug(f"Initializing LLM - Provider: {provider}, Model: {model_name}")

        if provider == "openai":
            # Validate OpenAI configuration
            if not self.settings.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY is not configured")
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

            try:
                api_key = self.settings.OPENAI_API_KEY.get_secret_value()

                # Create OpenAI model with specified parameters
                model = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key,
                )

                logger.debug("OpenAI ChatModel initialized successfully")
                return model

            except Exception as e:
                logger.error(f"Failed to initialize OpenAI model: {e}")
                raise ValueError(f"Failed to initialize OpenAI model: {e}") from e
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _format_context(self, context_chunks: List[str]) -> str:
        """
        Format context chunks into a single string for the prompt.

        Args:
            context_chunks: List of text chunks to use as context

        Returns:
            str: Formatted context string with separators
        """
        if not context_chunks:
            logger.debug("No context chunks provided")
            return "No context provided."

        # Join chunks with separators for clarity
        formatted = "\n---\n".join(context_chunks)
        logger.debug(
            f"Formatted {len(context_chunks)} context chunks into {len(formatted)} characters"
        )
        return formatted

    async def generate_answer(self, request: GenerateRequest) -> str:
        """
        Generate an answer using the LLM chain.

        Args:
            request: Generate request containing query and context

        Returns:
            str: Generated answer from the LLM

        Raises:
            HTTPException: If LLM invocation fails
        """
        query_preview = (
            request.query[:50] + "..." if len(request.query) > 50 else request.query
        )
        logger.info(f"Generating answer for query: '{query_preview}'")
        logger.debug(f"Processing {len(request.context_chunks)} context chunks")

        # Format context for the prompt
        formatted_context = self._format_context(request.context_chunks)

        # Prepare chain input
        chain_input = {"context": formatted_context, "query": request.query}

        try:
            logger.debug("Invoking RAG chain...")
            result = await self.rag_chain.ainvoke(chain_input)

            # Log response details
            logger.info(
                f"LLM response generated successfully (length: {len(result)} chars)"
            )
            if logger.isEnabledFor(logging.DEBUG):
                preview = result[:100] + "..." if len(result) > 100 else result
                logger.debug(f"Response preview: '{preview}'")

            return result

        except Exception as e:
            logger.error(f"Error invoking LLM chain: {e}", exc_info=True)

            # Provide specific error messages based on exception content
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                detail = "LLM service rate limit exceeded. Please try again later."
            elif any(
                term in error_msg
                for term in ["authentication", "unauthorized", "api key"]
            ):
                detail = (
                    "LLM service authentication failed. Please check configuration."
                )
            elif "timeout" in error_msg:
                detail = "LLM service request timed out. Please try again."
            else:
                detail = f"Failed to get response from LLM: {str(e)}"

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail,
            ) from e

    def is_healthy(self) -> bool:
        """
        Check if the service is properly initialized and healthy.

        Returns:
            bool: True if all components are initialized, False otherwise
        """
        try:
            # Verify all critical components are present
            health_checks = [
                self.rag_chain is not None,
                self.chat_model is not None,
                self.prompt_template is not None,
                self.output_parser is not None,
            ]

            is_healthy = all(health_checks)
            logger.debug(f"Health check result: {is_healthy}")
            return is_healthy

        except Exception as e:
            logger.warning(f"Health check failed with exception: {e}")
            return False
