# app/services/http_client.py
import httpx
from typing import Optional, Dict, Any, Union, List
from contextlib import asynccontextmanager
from fastapi import HTTPException, status

# Global httpx client instance (managed by lifespan)
_client: Optional[httpx.AsyncClient] = None

async def get_http_client() -> httpx.AsyncClient:
    """
    Dependency function to get the shared httpx.AsyncClient instance.
    Relies on the lifespan manager to initialize and close the client.
    """
    if _client is None:
        # This should ideally not happen if lifespan is managed correctly
        # but serves as a fallback or indicates a setup issue.
        raise RuntimeError("HTTP Client not initialized. Check FastAPI lifespan management.")
    return _client

@asynccontextmanager
async def lifespan_http_client(app, timeout: float):
    """
    Async context manager for managing the lifespan of the HTTP client.
    To be used with FastAPI's lifespan event handler.
    """
    global _client
    print("Initializing HTTP client...")
    _client = httpx.AsyncClient(timeout=timeout)
    try:
        yield # Application runs here
    finally:
        print("Closing HTTP client...")
        await _client.aclose()
        _client = None # Clear the global instance


async def make_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    json_data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], List[Any]]:
    """
    Makes an asynchronous HTTP request and handles common errors.

    Args:
        client: The httpx.AsyncClient instance.
        method: HTTP method (e.g., "GET", "POST").
        url: The URL to request.
        json_data: The JSON payload for POST/PUT requests.
        params: URL query parameters.

    Returns:
        The JSON response from the service.

    Raises:
        HTTPException: If the request fails or the downstream service returns an error.
    """
    try:
        response = await client.request(method, url, json=json_data, params=params)
        response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
        return response.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request timed out when calling {url}",
        )
    except httpx.RequestError as exc:
        # Includes connection errors, invalid URL errors, etc.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error connecting to {url}: {exc}",
        )
    except httpx.HTTPStatusError as exc:
        # Handle specific errors from the downstream service
        raise HTTPException(
            status_code=exc.response.status_code, # Propagate status code
            detail=f"Downstream service at {url} returned error: {exc.response.text}",
        )