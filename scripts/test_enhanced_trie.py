import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trie import AutocompleteTrie


def test_enhanced_trie():
    """Test enhanced Trie functionality"""
    print("🧪 Testing Enhanced Trie Smart Search Functionality")
    print("=" * 50)

    trie = AutocompleteTrie("models/autocomplete/enhanced_trie_data.pkl")

    test_cases = [
        {
            "query": "weather",
            "expected": "Should match weather-related queries",
            "description": "Chinese partial word matching test",
        },
        {
            "query": "python",
            "expected": "Should match Python-related queries",
            "description": "English partial word matching test",
        },
        {
            "query": "restaurant",
            "expected": "Should match restaurant-related queries",
            "description": "English exact matching test",
        },
        {
            "query": "email",
            "expected": "Should match email-related queries",
            "description": "English word matching test",
        },
        {
            "query": "pythn",
            "expected": "Should fuzzy match to python",
            "description": "Fuzzy matching test",
        },
        {
            "query": "wether",
            "expected": "Should fuzzy match to weather",
            "description": "English fuzzy matching test",
        },
        {
            "query": "learning",
            "expected": "Should match learning-related queries",
            "description": "Learning scenario test",
        },
        {
            "query": "schedule",
            "expected": "Should match schedule-related queries",
            "description": "English schedule scenario test",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        description = test_case["description"]
        expected = test_case["expected"]

        print(f"\n{i}. {description}")
        print(f"   Query: '{query}'")
        print(f"   Expected: {expected}")

        suggestions = trie.smart_search(query, 5)

        if suggestions:
            print(f"   Result: Found {len(suggestions)} matches")
            for j, suggestion in enumerate(suggestions, 1):
                original = suggestion.get("original_word", suggestion["word"])
                match_type = suggestion.get("match_type", "unknown")
                frequency = suggestion["frequency"]
                print(f"     {j}. {original}")
                print(f"        (Frequency: {frequency}, Match Type: {match_type})")
        else:
            print("   Result: ❌ No matches found")

    print("\n" + "=" * 50)
    print("📈 Performance Testing...")

    performance_queries = [
        "weather",
        "python",
        "learning",
        "development",
        "tutorial",
        "search",
    ]

    start_time = time.time()
    for query in performance_queries * 100:
        trie.smart_search(query, 10)
    end_time = time.time()

    avg_time = (end_time - start_time) / (len(performance_queries) * 100) * 1000
    print(f"Average query time: {avg_time:.2f} ms")

    print("\n✅ Enhanced Trie testing completed!")


if __name__ == "__main__":
    test_enhanced_trie()
