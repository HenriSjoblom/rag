import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, status

logger = logging.getLogger(__name__)

# This module's global HTTP client instance
_http_client_instance: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan_http_client(app: FastAPI, timeout: float):
    """
    Manages the lifecycle of the global httpx.AsyncClient instance.
    Initializes it on startup and closes it on shutdown.
    """
    global _http_client_instance
    if _http_client_instance is not None:
        logger.warning(
            "HTTP client instance already exists during lifespan startup. This might indicate multiple initializations."
        )

    logger.info(f"Initializing global HTTP client with timeout: {timeout}s")
    _http_client_instance = httpx.AsyncClient(timeout=timeout)
    try:
        yield
    finally:
        if _http_client_instance:
            logger.info("Closing global HTTP client session.")
            await _http_client_instance.aclose()
            _http_client_instance = None  # Clear the instance after closing
            logger.info("Global HTTP client session closed and instance cleared.")
        else:
            logger.warning("Attempted to close HTTP client, but no instance was found.")


def get_global_http_client() -> httpx.AsyncClient:
    """
    Returns the globally managed httpx.AsyncClient instance.
    Raises a RuntimeError if the client has not been initialized (e.g., lifespan did not run).
    """
    if _http_client_instance is None:
        logger.error(
            "Attempted to get HTTP client, but it's not initialized. Lifespan manager might not have run or has already shut down."
        )
        raise RuntimeError(
            "Global HTTP client is not initialized. Ensure the application lifespan manager has run."
        )
    return _http_client_instance


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
        response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
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
            status_code=exc.response.status_code,  # Propagate status code
            detail=f"Downstream service at {url} returned error: {exc.response.text}",
        )
