from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for initiating a chat interaction."""

    user_id: str = Field(..., description="Unique identifier for the user session.")
    message: str = Field(..., min_length=1, description="The user's message.")


class RetrievalRequest(BaseModel):
    """Data sent TO the retrieval service."""

    query: str


class RetrievalResponse(BaseModel):
    """Data received FROM the retrieval service."""

    chunks: List[str]


class GenerationRequest(BaseModel):
    """Data sent TO the generation service."""

    query: str
    context_chunks: List[str]


class GenerationResponse(BaseModel):
    """Data received FROM the generation service."""

    answer: str


class ChatResponse(BaseModel):
    """Response model sent back to the client."""

    response: str


class ErrorDetail(BaseModel):
    type: str
    loc: List[str | int]
    msg: str
    input: Any
    url: Optional[str] = None


class HTTPValidationError(BaseModel):
    detail: List[ErrorDetail]


class ServiceErrorResponse(BaseModel):
    detail: str


class IngestionUploadResponse(BaseModel):
    status: str
    documents_found: Optional[int] = None
    message: str


class IngestionDeleteResponse(BaseModel):
    message: str
    details: Optional[List[str]] = None
    files_deleted_count: Optional[int] = None
    collection_deleted: Optional[bool] = None
    source_files_cleared: Optional[bool] = None


class IngestionStatusResponse(BaseModel):
    is_processing: bool
    status: str
    last_completed: Optional[str] = None
    documents_processed: Optional[int] = None
    chunks_added: Optional[int] = None
    errors: Optional[List[str]] = None
    completion_time: Optional[str] = None


# Models for listing documents via RAG service
class RagDocumentDetail(BaseModel):
    name: str


class RagDocumentListResponse(BaseModel):
    count: int
    documents: List[RagDocumentDetail]
