#!/usr/bin/env python3
"""
Enhanced Trie initialization script
Supports Chinese word segmentation, fuzzy matching and smart search
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trie import AutocompleteTrie


def main():
    """Main function"""
    trie = AutocompleteTrie("models/autocomplete/enhanced_trie_data.pkl")
    data_file = "data/search_queries.txt"

    trie.load_from_text_file(data_file)
    trie.save_to_disk()

    stats = trie.get_stats()
    print(f"Total words: {stats['total_words']}")
    print(f"Total frequency: {stats['total_frequency']}")
    print(f"Storage location: {stats['persistence_file']}")

    test_cases = [
        "weather",
        "python",
        "restaurant",
        "email",
        "天气",
        "餐厅",
    ]

    for test_query in test_cases:
        suggestions = trie.smart_search(test_query, 5)
        if suggestions:
            print(f"'{test_query}': {len(suggestions)} matches")

    top_queries = trie.get_top_queries(10)
    print(f"Top {len(top_queries)} queries loaded")
    print(f"Trie saved to: {trie.persistence_file}")


if __name__ == "__main__":
    main()
