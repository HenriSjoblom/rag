# api_gateway/src/api/routes.py
from fastapi import APIRouter, HTTPException

import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/query")
async def query():
  try:
    return {"response": "This is a test query response"}
  except Exception as e:
    logger.error(f"Error in query endpoint: {str(e)}")
    raise HTTPException(status_code=500, detail="Internal server error")