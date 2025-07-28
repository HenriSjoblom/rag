"""
Minimal test to check if pytest works at all.
"""


def test_basic_assertion():
    """Test basic assertion works."""
    assert 1 + 1 == 2


def test_basic_math():
    """Test basic math operations."""
    assert 2 * 3 == 6
    assert 10 / 2 == 5


def test_string_operations():
    """Test string operations."""
    text = "hello world"
    assert text.upper() == "HELLO WORLD"
    assert len(text) == 11
