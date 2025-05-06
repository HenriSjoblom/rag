from pydantic import BaseModel, Field
from typing import Optional

class IngestionResponse(BaseModel):
    """
    Response model indicating the status of the ingestion task.
    """
    status: str = Field(..., description="Status message, e.g., 'Ingestion task started'.")
    documents_found: Optional[int] = Field(None, description="Number of documents found in the source directory.")
    message: Optional[str] = Field(None, description="Additional details or confirmation.")

class IngestionStatus(BaseModel):
    """
    Model to potentially track progress.
    For now, just used internally for results.
    """
    documents_processed: int = 0
    chunks_added: int = 0
    errors: list[str] = []