"""
Test script for the autocomplete trie functionality.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trie import AutocompleteTrie
import tempfile


def test_trie_basic_functionality():
    """Test basic trie operations."""
    print("Testing basic trie functionality...")

    # Create a temporary trie
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
        tmp_path = tmp_file.name

    try:
        trie = AutocompleteTrie(persistence_file=tmp_path)

        # Test insertion
        test_words = [
            "hello",
            "hello world",
            "help",
            "hero",
            "heroine",
            "python",
            "programming",
        ]
        for word in test_words:
            trie.insert(word)

        # Test search
        assert trie.search("hello") == True
        assert trie.search("hell") == False  # Not a complete word
        assert trie.search("hero") == True
        assert trie.search("xyz") == False

        # Test suggestions
        suggestions = trie.get_suggestions("he", max_suggestions=5)
        suggestion_words = [s["word"] for s in suggestions]
        assert "hello" in suggestion_words
        assert "help" in suggestion_words
        assert "hero" in suggestion_words

        # Test frequency update
        trie.update_frequency("hello", 5)
        suggestions = trie.get_suggestions("he", max_suggestions=5)
        # "hello" should be first due to higher frequency
        assert suggestions[0]["word"] == "hello"
        assert suggestions[0]["frequency"] >= 5

        # Test persistence
        trie.save_to_disk()

        # Create new trie and load from disk
        trie2 = AutocompleteTrie(persistence_file=tmp_path)
        assert trie2.search("hello") == True
        suggestions2 = trie2.get_suggestions("he", max_suggestions=5)
        assert len(suggestions2) > 0

        # Test stats
        stats = trie.get_stats()
        assert stats["total_words"] == len(test_words)

        print("✅ All basic tests passed!")

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_trie_with_file():
    """Test loading from text file."""
    print("Testing file loading...")

    # Create a test file
    test_data = "machine learning\ndata science\npython programming\nweb development\n"
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(test_data)
        tmp_file_path = tmp_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_trie:
        tmp_trie_path = tmp_trie.name

    try:
        trie = AutocompleteTrie(persistence_file=tmp_trie_path)
        trie.load_from_text_file(tmp_file_path)

        # Verify the data was loaded
        assert trie.search("machine learning") == True
        assert trie.search("data science") == True
        assert trie.search("python programming") == True

        # Test suggestions
        suggestions = trie.get_suggestions("data", max_suggestions=5)
        suggestion_words = [s["word"] for s in suggestions]
        assert "data science" in suggestion_words

        print("✅ File loading test passed!")

    finally:
        # Cleanup
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        if os.path.exists(tmp_trie_path):
            os.unlink(tmp_trie_path)


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
        tmp_path = tmp_file.name

    try:
        trie = AutocompleteTrie(persistence_file=tmp_path)

        # Test empty string
        suggestions = trie.get_suggestions("", max_suggestions=5)
        assert len(suggestions) == 0

        # Test non-existent prefix
        suggestions = trie.get_suggestions("xyz123", max_suggestions=5)
        assert len(suggestions) == 0

        # Test with special characters
        trie.insert("hello-world")
        trie.insert("hello_world")
        suggestions = trie.get_suggestions("hello", max_suggestions=5)
        suggestion_words = [s["word"] for s in suggestions]
        assert "hello-world" in suggestion_words or "hello_world" in suggestion_words

        print("✅ Edge cases test passed!")

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    """Run all tests."""
    print("🧪 Running autocomplete trie tests...\n")

    try:
        test_trie_basic_functionality()
        test_trie_with_file()
        test_edge_cases()
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
