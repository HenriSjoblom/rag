"""
Unit tests for health router endpoints in the RAG service.
"""

import pytest

from app.routers.health import health_check


class TestHealthCheck:
    """Test cases for health_check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        # Execute
        result = await health_check()

        # Verify
        assert result == {"status": "ok"}
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_response_structure(self):
        """Test that health check returns correct response structure."""
        result = await health_check()

        # Verify response structure
        assert isinstance(result, dict)
        assert len(result) == 1
        assert list(result.keys()) == ["status"]
        assert isinstance(result["status"], str)

    @pytest.mark.asyncio
    async def test_health_check_multiple_calls(self):
        """Test that health check is consistent across multiple calls."""
        # Execute multiple times
        results = []
        for _ in range(5):
            result = await health_check()
            results.append(result)

        # Verify all results are the same
        for result in results:
            assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_check_is_async(self):
        """Test that health check function is properly async."""
        import asyncio
        import inspect

        # Verify function is async
        assert inspect.iscoroutinefunction(health_check)

        # Verify it returns a coroutine when called
        coro = health_check()
        assert inspect.iscoroutine(coro)

        # Execute and clean up
        result = await coro
        assert result == {"status": "ok"}
