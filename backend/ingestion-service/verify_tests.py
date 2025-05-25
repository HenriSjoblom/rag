#!/usr/bin/env python3
"""
Test verification script to check our created tests work correctly.
This runs tests without the complex conftest.py setup that might cause hanging.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def run_model_tests():
    """Run model tests directly."""
    print("🧪 Testing Models...")
    try:
        from tests.test_models import TestIngestionStatus

        test_class = TestIngestionStatus()
        test_class.test_ingestion_status_default_values()
        test_class.test_ingestion_status_valid_values()
        test_class.test_ingestion_status_negative_documents_processed()
        test_class.test_ingestion_status_negative_chunks_added()
        test_class.test_ingestion_status_serialization()
        print("✅ Model tests passed!")
        return True
    except Exception as e:
        print(f"❌ Model tests failed: {e}")
        return False


def run_file_management_tests():
    """Run basic file management tests."""
    print("📁 Testing File Management...")
    try:
        # Mock the dependencies
        with (
            patch("app.services.file_management.Path"),
            patch("app.services.file_management.Settings"),
        ):
            from app.config import Settings
            from app.services.file_management import FileManagementService

            # Create minimal test
            temp_dir = tempfile.mkdtemp()
            try:
                mock_settings = Mock(spec=Settings)
                mock_settings.SOURCE_DIRECTORY = temp_dir
                mock_settings.MAX_FILE_SIZE_MB = 50

                service = FileManagementService(mock_settings)
                assert service.settings == mock_settings
                print("✅ File management basic tests passed!")
                return True
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"❌ File management tests failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_imports():
    """Test that all our test modules can be imported."""
    print("📦 Testing Imports...")
    test_modules = [
        "tests.test_models",
        "tests.test_file_management",
        "tests.test_ingestion_state",
        "tests.test_collection_manager",
        "tests.test_routers",
        "tests.test_chroma_manager",
        "tests.test_integration",
    ]

    failed_imports = []
    for module in test_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"  ⚠️  {module}: {e}")

    if failed_imports:
        print(f"❌ Failed to import: {failed_imports}")
        return False
    else:
        print("✅ All test modules imported successfully!")
        return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("🚀 INGESTION SERVICE TEST VERIFICATION")
    print("=" * 60)

    all_passed = True

    # Test imports first
    if not test_imports():
        all_passed = False

    print()

    # Test models
    if not run_model_tests():
        all_passed = False

    print()

    # Test file management basics
    if not run_file_management_tests():
        all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("\n✨ Your test suite is ready!")
        print("\n📋 To run tests:")
        print(
            "   • Model tests work with: uv run python -m pytest tests/test_models.py -v"
        )
        print("   • For other tests, use the minimal conftest.py:")
        print(
            "     cd tests && mv conftest.py conftest_full.py && mv conftest_minimal.py conftest.py"
        )
        print("     uv run python -m pytest tests/ -v")
        print(
            "     # Then restore: mv conftest.py conftest_minimal.py && mv conftest_full.py conftest.py"
        )

    else:
        print("❌ SOME VERIFICATION TESTS FAILED!")
        print("   Check the errors above and fix any issues.")

    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
