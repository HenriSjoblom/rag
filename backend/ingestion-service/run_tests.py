#!/usr/bin/env python3
"""Simple test runner to check if our tests work."""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_models():
    """Test the models functionality."""
    print("Testing models...")

    try:
        from app.models import (
            DocumentDetail,
            DocumentListResponse,
            IngestionResponse,
            IngestionStatus,
            IngestionStatusResponse,
        )
        from pydantic import ValidationError

        # Test IngestionStatus
        print("  ✓ IngestionStatus import successful")
        status = IngestionStatus()
        assert status.documents_processed == 0
        assert status.chunks_added == 0
        assert status.errors == []
        print("  ✓ IngestionStatus default values work")

        # Test validation
        try:
            IngestionStatus(documents_processed=-1)
            print("  ✗ Validation should have failed")
            return False
        except ValidationError:
            print("  ✓ IngestionStatus validation works")

        # Test IngestionResponse
        response = IngestionResponse(status="completed")
        assert response.status == "completed"
        print("  ✓ IngestionResponse works")

        # Test DocumentDetail
        doc = DocumentDetail(name="test.pdf")
        assert doc.name == "test.pdf"
        print("  ✓ DocumentDetail works")

        # Test DocumentListResponse
        docs = [DocumentDetail(name="doc1.pdf")]
        list_response = DocumentListResponse(count=1, documents=docs)
        assert list_response.count == 1
        print("  ✓ DocumentListResponse works")

        # Test IngestionStatusResponse
        status_response = IngestionStatusResponse(is_processing=True, status="running")
        assert status_response.is_processing is True
        print("  ✓ IngestionStatusResponse works")

        # Test serialization
        status_with_data = IngestionStatus(
            documents_processed=3, chunks_added=50, errors=["Test error"]
        )
        data = status_with_data.model_dump()
        assert data["documents_processed"] == 3
        print("  ✓ IngestionStatus serialization works")

        print("All model tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_imports():
    """Test if we can import test modules."""
    print("Testing imports...")

    try:
        from tests.test_models import TestIngestionStatus

        print("  ✓ test_models imports successfully")

        # Try to instantiate and run a simple test
        test_instance = TestIngestionStatus()
        test_instance.test_ingestion_status_default_values()
        print("  ✓ test_ingestion_status_default_values runs successfully")

        return True

    except Exception as e:
        print(f"  ✗ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING INGESTION SERVICE")
    print("=" * 50)

    success = True

    if not test_models():
        success = False

    print()

    if not test_imports():
        success = False

    print()
    print("=" * 50)
    if success:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("=" * 50)
