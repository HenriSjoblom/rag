from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class IngestionStatus(BaseModel):
    """Model for tracking ingestion results."""

    documents_processed: int = Field(default=0, ge=0)
    chunks_added: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)


class IngestionResponse(BaseModel):
    """Response model for ingestion operations."""

    status: str = Field(..., min_length=1)
    documents_found: Optional[int] = Field(None, ge=0)
    message: Optional[str] = None


class DocumentDetail(BaseModel):
    """Details for a single document."""

    name: str = Field(..., min_length=1)


class DocumentListResponse(BaseModel):
    """Response for document listing operations."""

    count: int = Field(..., ge=0)
    documents: List[DocumentDetail]

    @model_validator(mode="after")
    def validate_count_matches_documents(self):
        if len(self.documents) != self.count:
            raise ValueError("Count must match number of documents")
        return self


class IngestionStatusResponse(BaseModel):
    """Response model for ingestion status checks."""

    is_processing: bool
    status: str = Field(..., min_length=1)
    last_completed: Optional[str] = None
    documents_processed: Optional[int] = Field(None, ge=0)
    chunks_added: Optional[int] = Field(None, ge=0)
    errors: Optional[List[str]] = None
