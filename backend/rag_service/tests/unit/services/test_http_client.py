"""
Unit tests for HTTP client utilities in the RAG service.
"""

import httpx
import pytest
from app.services.http_client import (
    get_global_http_client,
    lifespan_http_client,
    make_request,
)
from fastapi import FastAPI, HTTPException



class TestMakeRequest:
    """Test cases for make_request function."""

    @pytest.mark.asyncio
    async def test_make_request_success_get(self, mock_http_client, mocker):
        """Test successful GET request."""
        # Setup mock response
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"status": "success"}
        mock_http_client.request.return_value = mock_response

        result = await make_request(
            client=mock_http_client,
            method="GET",
            url="http://test-service/api/endpoint"
        )

        mock_http_client.request.assert_called_once_with(
            "GET", 
            "http://test-service/api/endpoint",
            json=None,
            params=None
        )
        mock_response.raise_for_status.assert_called_once()
        assert result == {"status": "success"}

    @pytest.mark.asyncio
    async def test_make_request_success_post_with_json(self, mock_http_client, mocker):
        """Test successful POST request with JSON data."""
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"id": 123, "message": "created"}
        mock_http_client.request.return_value = mock_response

        json_data = {"name": "test", "value": 42}
        result = await make_request(
            client=mock_http_client,
            method="POST",
            url="http://test-service/api/create",
            json_data=json_data
        )

        mock_http_client.request.assert_called_once_with(
            "POST",
            "http://test-service/api/create",
            json=json_data,
            params=None
        )
        assert result == {"id": 123, "message": "created"}

    @pytest.mark.asyncio
    async def test_make_request_with_params(self, mock_http_client, mocker):
        """Test request with URL parameters."""
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_http_client.request.return_value = mock_response

        params = {"page": 1, "limit": 10}
        await make_request(
            client=mock_http_client,
            method="GET",
            url="http://test-service/api/search",
            params=params
        )

        mock_http_client.request.assert_called_once_with(
            "GET",
            "http://test-service/api/search",
            json=None,
            params=params
        )

    @pytest.mark.asyncio
    async def test_make_request_timeout_exception(self, mock_http_client, mocker):
        """Test handling of timeout exceptions."""
        mock_http_client.request.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(HTTPException) as exc_info:
            await make_request(
                client=mock_http_client,
                method="GET",
                url="http://test-service/api/slow"
            )

        assert exc_info.value.status_code == 504
        assert "Request timed out" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_make_request_connection_error(self, mock_http_client, mocker):
        """Test handling of connection errors."""
        mock_http_client.request.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await make_request(
                client=mock_http_client,
                method="GET",
                url="http://unreachable-service/api/endpoint"
            )

        assert exc_info.value.status_code == 503
        assert "Error connecting" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_make_request_http_status_error(self, mock_http_client, mocker):
        """Test handling of HTTP status errors."""
        mock_response = mocker.MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        
        http_status_error = httpx.HTTPStatusError(
            "404 Not Found",
            request=mocker.MagicMock(),
            response=mock_response
        )
        mock_http_client.request.side_effect = http_status_error

        with pytest.raises(HTTPException) as exc_info:
            await make_request(
                client=mock_http_client,
                method="GET",
                url="http://test-service/api/missing"
            )

        assert exc_info.value.status_code == 404
        assert "Downstream service" in exc_info.value.detail
        assert "Not Found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_make_request_general_request_error(self, mock_http_client, mocker):
        """Test handling of general request errors."""
        mock_http_client.request.side_effect = httpx.RequestError("Invalid URL")

        with pytest.raises(HTTPException) as exc_info:
            await make_request(
                client=mock_http_client,
                method="GET",
                url="http://invalid-url"
            )

        assert exc_info.value.status_code == 503
        assert "Error connecting" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_make_request_returns_list(self, mock_http_client, mocker):
        """Test that make_request can return list responses."""
        mock_response = mocker.MagicMock()
        mock_response.json.return_value = [{"id": 1}, {"id": 2}]
        mock_http_client.request.return_value = mock_response

        result = await make_request(
            client=mock_http_client,
            method="GET",
            url="http://test-service/api/list"
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == 1


class TestLifespanHttpClient:
    """Test cases for lifespan_http_client context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_http_client_initialization(self, mocker):
        """Test that lifespan properly initializes and closes HTTP client."""
        mock_app = mocker.MagicMock(spec=FastAPI)
        mock_client = mocker.AsyncMock(spec=httpx.AsyncClient)
        
        # Mock the AsyncClient constructor
        mock_async_client_class = mocker.patch(
            "app.services.http_client.httpx.AsyncClient",
            return_value=mock_client
        )

        async with lifespan_http_client(mock_app, timeout=30.0):
            # During the lifespan context, client should be initialized
            mock_async_client_class.assert_called_once_with(timeout=30.0)

        # After context exits, client should be closed
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_http_client_existing_instance_warning(self, mocker, caplog):
        """Test warning when client instance already exists."""
        mock_app = mocker.MagicMock(spec=FastAPI)
        mock_client = mocker.AsyncMock(spec=httpx.AsyncClient)
        
        # Patch the global variable to simulate existing instance
        mocker.patch("app.services.http_client._http_client_instance", mock_client)
        mocker.patch("app.services.http_client.httpx.AsyncClient", return_value=mock_client)

        async with lifespan_http_client(mock_app, timeout=30.0):
            pass

        assert "already exists during lifespan startup" in caplog.text

    @pytest.mark.asyncio
    async def test_lifespan_http_client_close_when_none(self, mocker, caplog):
        """Test warning when trying to close non-existent client."""
        mock_app = mocker.MagicMock(spec=FastAPI)
        
        # Mock to return None when trying to close
        mocker.patch("app.services.http_client._http_client_instance", None)
        mocker.patch("app.services.http_client.httpx.AsyncClient")

        async with lifespan_http_client(mock_app, timeout=30.0):
            # Simulate the instance being None during cleanup
            mocker.patch("app.services.http_client._http_client_instance", None)

        assert "no instance was found" in caplog.text


class TestGetGlobalHttpClient:
    """Test cases for get_global_http_client function."""

    def test_get_global_http_client_success(self, mocker):
        """Test successful retrieval of global HTTP client."""
        mock_client = mocker.MagicMock(spec=httpx.AsyncClient)
        mocker.patch("app.services.http_client._http_client_instance", mock_client)

        result = get_global_http_client()
        assert result == mock_client

    def test_get_global_http_client_not_initialized(self, mocker):
        """Test error when global HTTP client is not initialized."""
        mocker.patch("app.services.http_client._http_client_instance", None)

        with pytest.raises(RuntimeError) as exc_info:
            get_global_http_client()

        assert "not initialized" in str(exc_info.value)
        assert "lifespan manager" in str(exc_info.value)
