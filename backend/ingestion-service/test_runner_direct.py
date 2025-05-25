#!/usr/bin/env python3
"""
Direct test execution without pytest to verify our tests work.
"""

import sys
import traceback
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def run_test_models():
    """Run the model tests directly."""
    print("=" * 60)
    print("RUNNING MODEL TESTS")
    print("=" * 60)

    try:
        # Import the test class
        from tests.test_models import (
            TestDocumentDetail,
            TestDocumentListResponse,
            TestIngestionResponse,
            TestIngestionStatus,
            TestIngestionStatusResponse,
        )

        # Test IngestionStatus
        print("\n--- Testing IngestionStatus ---")
        test_ing_status = TestIngestionStatus()

        try:
            test_ing_status.test_ingestion_status_default_values()
            print("✓ test_ingestion_status_default_values")
        except Exception as e:
            print(f"✗ test_ingestion_status_default_values: {e}")

        try:
            test_ing_status.test_ingestion_status_valid_values()
            print("✓ test_ingestion_status_valid_values")
        except Exception as e:
            print(f"✗ test_ingestion_status_valid_values: {e}")

        try:
            test_ing_status.test_ingestion_status_negative_documents_processed()
            print("✓ test_ingestion_status_negative_documents_processed")
        except Exception as e:
            print(f"✗ test_ingestion_status_negative_documents_processed: {e}")

        try:
            test_ing_status.test_ingestion_status_negative_chunks_added()
            print("✓ test_ingestion_status_negative_chunks_added")
        except Exception as e:
            print(f"✗ test_ingestion_status_negative_chunks_added: {e}")

        try:
            test_ing_status.test_ingestion_status_serialization()
            print("✓ test_ingestion_status_serialization")
        except Exception as e:
            print(f"✗ test_ingestion_status_serialization: {e}")

        # Test IngestionResponse
        print("\n--- Testing IngestionResponse ---")
        test_ing_response = TestIngestionResponse()

        try:
            test_ing_response.test_ingestion_response_required_fields()
            print("✓ test_ingestion_response_required_fields")
        except Exception as e:
            print(f"✗ test_ingestion_response_required_fields: {e}")

        try:
            test_ing_response.test_ingestion_response_all_fields()
            print("✓ test_ingestion_response_all_fields")
        except Exception as e:
            print(f"✗ test_ingestion_response_all_fields: {e}")

        try:
            test_ing_response.test_ingestion_response_empty_status()
            print("✓ test_ingestion_response_empty_status")
        except Exception as e:
            print(f"✗ test_ingestion_response_empty_status: {e}")

        try:
            test_ing_response.test_ingestion_response_negative_documents_found()
            print("✓ test_ingestion_response_negative_documents_found")
        except Exception as e:
            print(f"✗ test_ingestion_response_negative_documents_found: {e}")

        # Test DocumentDetail
        print("\n--- Testing DocumentDetail ---")
        test_doc_detail = TestDocumentDetail()

        try:
            test_doc_detail.test_document_detail_valid_name()
            print("✓ test_document_detail_valid_name")
        except Exception as e:
            print(f"✗ test_document_detail_valid_name: {e}")

        try:
            test_doc_detail.test_document_detail_empty_name()
            print("✓ test_document_detail_empty_name")
        except Exception as e:
            print(f"✗ test_document_detail_empty_name: {e}")

        # Test DocumentListResponse
        print("\n--- Testing DocumentListResponse ---")
        test_doc_list = TestDocumentListResponse()

        try:
            test_doc_list.test_document_list_response_valid()
            print("✓ test_document_list_response_valid")
        except Exception as e:
            print(f"✗ test_document_list_response_valid: {e}")

        try:
            test_doc_list.test_document_list_response_empty_list()
            print("✓ test_document_list_response_empty_list")
        except Exception as e:
            print(f"✗ test_document_list_response_empty_list: {e}")

        try:
            test_doc_list.test_document_list_response_count_mismatch()
            print("✓ test_document_list_response_count_mismatch")
        except Exception as e:
            print(f"✗ test_document_list_response_count_mismatch: {e}")

        try:
            test_doc_list.test_document_list_response_negative_count()
            print("✓ test_document_list_response_negative_count")
        except Exception as e:
            print(f"✗ test_document_list_response_negative_count: {e}")

        # Test IngestionStatusResponse
        print("\n--- Testing IngestionStatusResponse ---")
        test_status_response = TestIngestionStatusResponse()

        try:
            test_status_response.test_ingestion_status_response_minimal()
            print("✓ test_ingestion_status_response_minimal")
        except Exception as e:
            print(f"✗ test_ingestion_status_response_minimal: {e}")

        try:
            test_status_response.test_ingestion_status_response_full()
            print("✓ test_ingestion_status_response_full")
        except Exception as e:
            print(f"✗ test_ingestion_status_response_full: {e}")

        try:
            test_status_response.test_ingestion_status_response_empty_status()
            print("✓ test_ingestion_status_response_empty_status")
        except Exception as e:
            print(f"✗ test_ingestion_status_response_empty_status: {e}")

        try:
            test_status_response.test_ingestion_status_response_negative_values()
            print("✓ test_ingestion_status_response_negative_values")
        except Exception as e:
            print(f"✗ test_ingestion_status_response_negative_values: {e}")

        try:
            test_status_response.test_ingestion_status_response_serialization()
            print("✓ test_ingestion_status_response_serialization")
        except Exception as e:
            print(f"✗ test_ingestion_status_response_serialization: {e}")

        print("\n" + "=" * 60)
        print("MODEL TESTS COMPLETED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"Error importing or running tests: {e}")
        traceback.print_exc()
        return False


def run_test_endpoints():
    """Run basic endpoint tests to debug dependency injection."""
    print("=" * 60)
    print("RUNNING ENDPOINT TESTS")
    print("=" * 60)

    try:
        import io
        from unittest.mock import AsyncMock, Mock

        from app.deps import (
            get_file_management_service,
            get_ingestion_state_service,
            get_settings,
        )
        from app.main import app
        from fastapi.testclient import TestClient

        # Create basic mocks
        mock_settings = Mock()
        mock_settings.chroma_host = "localhost"
        mock_settings.chroma_port = 8000
        mock_settings.documents_dir = "/tmp/test_docs"
        mock_settings.collection_name = "test_collection"

        # Create async mock for state service
        mock_state_service = AsyncMock()
        mock_state_service.get_status.return_value = {
            "is_processing": False,
            "last_completed": None,
            "current_status": "idle",
            "documents_processed": 0,
            "chunks_added": 0,
            "error_message": None,
        }
        mock_state_service.is_processing.return_value = False

        # Create regular mock for file service
        mock_file_service = Mock()
        mock_file_service.list_documents.return_value = [
            {"name": "test1.pdf", "size": 1000},
            {"name": "test2.pdf", "size": 2000},
        ]
        mock_file_service.count_documents.return_value = 2
        mock_file_service.save_document.return_value = {
            "name": "uploaded.pdf",
            "size": 1500,
            "message": "File uploaded successfully",
        }
        mock_file_service.has_duplicate_filename.return_value = False

        # Create collection service mock
        mock_collection_service = Mock()
        mock_collection_service.clear_collection_and_documents.return_value = {
            "collection_cleared": True,
            "documents_cleared": True,
            "messages": [
                "Collection cleared successfully",
                "Documents cleared successfully",
            ],
        }

        # Create processor mock
        mock_processor = Mock()
        mock_processor.run_ingestion.return_value = {
            "status": "completed",
            "documents_processed": 2,
            "chunks_added": 50,
        }

        # Override dependencies
        app.dependency_overrides[get_settings] = lambda: mock_settings
        app.dependency_overrides[get_ingestion_state_service] = (
            lambda request: mock_state_service
        )
        app.dependency_overrides[get_file_management_service] = (
            lambda **kwargs: mock_file_service
        )

        # Add overrides for other dependencies that might be missing
        try:
            from app.deps import (
                get_collection_management_service,
                get_ingestion_processor,
            )

            app.dependency_overrides[get_collection_management_service] = (
                lambda **kwargs: mock_collection_service
            )
            app.dependency_overrides[get_ingestion_processor] = (
                lambda **kwargs: mock_processor
            )
        except ImportError:
            print("⚠️  Warning: Some dependency functions not found, tests may fail")

        # CRITICAL FIX: Override the app.state directly for status endpoint
        # This is what the actual app.main.py uses
        print("\n--- Overriding app.state for direct service access ---")
        if not hasattr(app.state, "ingestion_state_service"):
            app.state.ingestion_state_service = mock_state_service
            print("✓ Set app.state.ingestion_state_service to AsyncMock")
        else:
            print(
                f"⚠️  app.state.ingestion_state_service already exists: {type(app.state.ingestion_state_service)}"
            )

        with TestClient(app) as client:
            print("\n--- Testing Health Endpoint ---")
            try:
                response = client.get("/health")
                print(f"Health endpoint: {response.status_code}")
                if response.status_code == 200:
                    print("✓ Health endpoint works")
                    print(f"Response: {response.json()}")
                else:
                    print(f"✗ Health endpoint failed: {response.text}")
            except Exception as e:
                print(f"✗ Health endpoint error: {e}")

            print("\n--- Testing Status Endpoint ---")
            try:
                response = client.get("/api/v1/status")
                print(f"Status endpoint: {response.status_code}")
                if response.status_code == 200:
                    print("✓ Status endpoint works")
                    print(f"Response: {response.json()}")
                else:
                    print(f"✗ Status endpoint failed: {response.text}")
                    if response.status_code == 422:
                        print(f"Validation errors: {response.json()}")
            except Exception as e:
                print(f"✗ Status endpoint error: {e}")
                import traceback

                traceback.print_exc()

            print("\n--- Testing Documents Endpoint ---")
            try:
                response = client.get("/api/v1/documents")
                print(f"Documents endpoint: {response.status_code}")
                if response.status_code == 200:
                    print("✓ Documents endpoint works")
                    print(f"Response: {response.json()}")
                elif response.status_code == 307:
                    # Try with trailing slash
                    response = client.get("/api/v1/documents/")
                    print(f"Documents endpoint (with slash): {response.status_code}")
                    if response.status_code == 200:
                        print("✓ Documents endpoint works with trailing slash")
                        print(f"Response: {response.json()}")
                    else:
                        print(f"✗ Documents endpoint failed: {response.text}")
                        if response.status_code == 422:
                            print(f"Validation errors: {response.json()}")
                else:
                    print(f"✗ Documents endpoint failed: {response.text}")
                    if response.status_code == 422:
                        print(f"Validation errors: {response.json()}")
            except Exception as e:
                print(f"✗ Documents endpoint error: {e}")

            print("\n--- Testing Ingest Endpoint ---")
            try:
                response = client.post("/api/v1/ingest")
                print(f"Ingest endpoint: {response.status_code}")
                if response.status_code == 200:
                    print("✓ Ingest endpoint works")
                    print(f"Response: {response.json()}")
                else:
                    print(f"✗ Ingest endpoint failed: {response.text}")
                    # Print validation errors if 422
                    if response.status_code == 422:
                        print(f"Validation errors: {response.json()}")
            except Exception as e:
                print(f"✗ Ingest endpoint error: {e}")

            print("\n--- Testing Upload Endpoint ---")
            try:
                test_content = b"This is a test PDF content"
                test_file = io.BytesIO(test_content)
                response = client.post(
                    "/api/v1/upload",
                    files={"file": ("test.pdf", test_file, "application/pdf")},
                )
                print(f"Upload endpoint: {response.status_code}")
                if response.status_code == 200:
                    print("✓ Upload endpoint works")
                    print(f"Response: {response.json()}")
                else:
                    print(f"✗ Upload endpoint failed: {response.text}")
                    # Print validation errors if 422
                    if response.status_code == 422:
                        print(f"Validation errors: {response.json()}")
            except Exception as e:
                print(f"✗ Upload endpoint error: {e}")

            print("\n--- Testing Collection Clear Endpoint ---")
            try:
                response = client.delete("/api/v1/collection")
                print(f"Collection clear endpoint: {response.status_code}")
                if response.status_code == 200:
                    print("✓ Collection clear endpoint works")
                    print(f"Response: {response.json()}")
                elif response.status_code == 307:
                    # Try with trailing slash
                    response = client.delete("/api/v1/collection/")
                    print(
                        f"Collection clear endpoint (with slash): {response.status_code}"
                    )
                    if response.status_code == 200:
                        print("✓ Collection clear endpoint works with trailing slash")
                        print(f"Response: {response.json()}")
                    else:
                        print(f"✗ Collection clear endpoint failed: {response.text}")
                        if response.status_code == 422:
                            print(f"Validation errors: {response.json()}")
                else:
                    print(f"✗ Collection clear endpoint failed: {response.text}")
                    # Print validation errors if 422
                    if response.status_code == 422:
                        print(f"Validation errors: {response.json()}")
            except Exception as e:
                print(f"✗ Collection clear endpoint error: {e}")

            print("\n--- Testing Available Routes ---")
            try:
                # List all available routes
                routes = []
                for route in app.routes:
                    if hasattr(route, "path"):
                        routes.append(
                            f"{route.methods if hasattr(route, 'methods') else 'N/A'} {route.path}"
                        )
                    elif hasattr(route, "prefix"):
                        # This is likely a router
                        for sub_route in route.routes:
                            if hasattr(sub_route, "path"):
                                full_path = route.prefix + sub_route.path
                                routes.append(
                                    f"{sub_route.methods if hasattr(sub_route, 'methods') else 'N/A'} {full_path}"
                                )

                print("Available routes:")
                for route in sorted(routes):
                    print(f"  {route}")

            except Exception as e:
                print(f"Error listing routes: {e}")

        # Clear overrides
        app.dependency_overrides.clear()

        print("\n" + "=" * 60)
        print("ENDPOINT TESTS COMPLETED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"Error running endpoint tests: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run both model tests and endpoint tests
    print("🚀 Starting comprehensive test execution...\n")

    model_success = run_test_models()
    endpoint_success = run_test_endpoints()

    if model_success and endpoint_success:
        print("\n🎉 All tests executed successfully!")
    else:
        print(
            f"\n❌ Test results - Models: {'✓' if model_success else '✗'}, Endpoints: {'✓' if endpoint_success else '✗'}"
        )
        sys.exit(1)
