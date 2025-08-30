"""
Trie data structure for autocomplete functionality.
Supports persistence, LRU cache, and frequency-based suggestions.
"""

import pickle
from typing import List, Optional, Dict
from functools import lru_cache
from collections import defaultdict
import os


class TrieNode:
    """A single node in the Trie data structure."""

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end_of_word: bool = False
        self.frequency: int = 0  # Track how often this word is used
        self.word: Optional[str] = None  # Store the complete word at end nodes


class AutocompleteTrie:
    """
    Trie data structure optimized for autocomplete functionality.
    Features:
    - Persistence to disk
    - Frequency-based ranking
    - LRU cache for fast lookups
    - Prefix-based suggestions
    """

    def __init__(self, persistence_file: str = "models/autocomplete/trie_data.pkl"):
        self.root = TrieNode()
        self.persistence_file = persistence_file
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self._create_data_dir()
        self.load_from_disk()

    def _create_data_dir(self):
        """Create data directory if it doesn't exist."""
        os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)

    def insert(self, word: str, frequency: int = 1):
        """
        Insert a word into the trie with optional frequency.

        Args:
            word (str): The word to insert
            frequency (int): The frequency/weight of this word
        """
        if not word or not word.strip():
            return

        word = word.strip().lower()
        node = self.root

        # Traverse the trie, creating nodes as needed
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        # Mark end of word and update frequency
        node.is_end_of_word = True
        node.word = word
        node.frequency += frequency
        self.word_frequencies[word] += frequency

    def insert_batch(self, words: List[str]):
        """
        Insert multiple words efficiently.

        Args:
            words (List[str]): List of words to insert
        """
        for word in words:
            self.insert(word)

    def load_from_text_file(self, file_path: str):
        """
        Load words from a text file (one word per line).

        Args:
            file_path (str): Path to the text file containing search queries
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                words = [line.strip() for line in f if line.strip()]
                self.insert_batch(words)
            print(f"Loaded {len(words)} queries from {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except (IOError, OSError) as e:
            print(f"Error loading from file: {e}")

    @lru_cache(maxsize=1000)
    def search(self, word: str) -> bool:
        """
        Check if a word exists in the trie.

        Args:
            word (str): The word to search for

        Returns:
            bool: True if word exists, False otherwise
        """
        word = word.strip().lower()
        node = self.root

        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_end_of_word

    @lru_cache(maxsize=500)
    def get_suggestions(
        self, prefix: str, max_suggestions: int = 10
    ) -> List[Dict[str, any]]:
        """
        Get autocomplete suggestions for a given prefix.

        Args:
            prefix (str): The prefix to search for
            max_suggestions (int): Maximum number of suggestions to return

        Returns:
            List[Dict]: List of suggestions with word and frequency
        """
        if not prefix or not prefix.strip():
            return []

        prefix = prefix.strip().lower()
        node = self.root

        # Navigate to the prefix node
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all words with this prefix
        suggestions = []
        self._collect_words(node, prefix, suggestions)

        # Sort by frequency (descending) and return top suggestions
        suggestions.sort(key=lambda x: x["frequency"], reverse=True)
        return suggestions[:max_suggestions]

    def _collect_words(
        self, node: TrieNode, current_word: str, suggestions: List[Dict[str, any]]
    ):
        """
        Recursively collect all words from a given node.

        Args:
            node (TrieNode): Current trie node
            current_word (str): Word built so far
            suggestions (List[Dict]): List to store suggestions
        """
        if node.is_end_of_word:
            suggestions.append({"word": current_word, "frequency": node.frequency})

        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, suggestions)

    def update_frequency(self, word: str, increment: int = 1):
        """
        Update the frequency of a word (e.g., when user selects a suggestion).

        Args:
            word (str): The word to update
            increment (int): Amount to increment frequency by
        """
        word = word.strip().lower()
        if self.search(word):
            node = self.root
            for char in word:
                node = node.children[char]
            node.frequency += increment
            self.word_frequencies[word] += increment

            # Clear cache to reflect updated frequencies
            self.get_suggestions.cache_clear()

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the trie.

        Returns:
            Dict: Statistics including total words, nodes, etc.
        """
        total_words = len(self.word_frequencies)
        total_frequency = sum(self.word_frequencies.values())

        return {
            "total_words": total_words,
            "total_frequency": total_frequency,
            "persistence_file": self.persistence_file,
        }

    def save_to_disk(self):
        """Save the trie to disk for persistence."""
        try:
            data = {
                "word_frequencies": dict(self.word_frequencies),
                "trie_structure": self._serialize_trie(),
            }
            with open(self.persistence_file, "wb") as f:
                pickle.dump(data, f)
            print(f"Trie saved to {self.persistence_file}")
        except (IOError, OSError, pickle.PickleError) as e:
            print(f"Error saving trie: {e}")

    def load_from_disk(self):
        """Load the trie from disk if it exists."""
        try:
            if (
                os.path.exists(self.persistence_file)
                and os.path.getsize(self.persistence_file) > 0
            ):
                with open(self.persistence_file, "rb") as f:
                    data = pickle.load(f)

                self.word_frequencies = defaultdict(
                    int, data.get("word_frequencies", {})
                )
                self._deserialize_trie(data.get("trie_structure", {}))
                print(f"Trie loaded from {self.persistence_file}")
            else:
                print(f"No existing trie file found at {self.persistence_file}")
        except (IOError, OSError, pickle.PickleError, EOFError) as e:
            print(f"Error loading trie: {e}")

    def _serialize_trie(self) -> Dict:
        """Serialize the trie structure to a dictionary."""

        def serialize_node(node: TrieNode) -> Dict:
            return {
                "is_end_of_word": node.is_end_of_word,
                "frequency": node.frequency,
                "word": node.word,
                "children": {
                    char: serialize_node(child) for char, child in node.children.items()
                },
            }

        return serialize_node(self.root)

    def _deserialize_trie(self, data: Dict):
        """Deserialize the trie structure from a dictionary."""

        def deserialize_node(node_data: Dict) -> TrieNode:
            node = TrieNode()
            node.is_end_of_word = node_data.get("is_end_of_word", False)
            node.frequency = node_data.get("frequency", 0)
            node.word = node_data.get("word")

            for char, child_data in node_data.get("children", {}).items():
                node.children[char] = deserialize_node(child_data)

            return node

        if data:
            self.root = deserialize_node(data)
        else:
            self.root = TrieNode()

    def clear_cache(self):
        """Clear all LRU caches."""
        self.search.cache_clear()
        self.get_suggestions.cache_clear()

    def get_top_queries(self, limit: int = 20) -> List[Dict[str, any]]:
        """
        Get the most frequent queries.

        Args:
            limit (int): Maximum number of queries to return

        Returns:
            List[Dict]: Top queries with their frequencies
        """
        sorted_words = sorted(
            self.word_frequencies.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {"word": word, "frequency": freq} for word, freq in sorted_words[:limit]
        ]


# Global trie instance
autocomplete_trie = AutocompleteTrie()
