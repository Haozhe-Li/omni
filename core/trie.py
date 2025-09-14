"""
Trie data structure for autocomplete functionality.
Supports persistence, LRU cache, frequency-based suggestions,
word segmentation, and fuzzy matching.
"""

import pickle
from typing import List, Optional, Dict
from functools import lru_cache
from collections import defaultdict
import os
import re
import jieba

STOP_WORDS = {
    "in",
    "out",
    "ing",
    "and",
    "or",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "by",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "must",
    "shall",
    "it",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "me",
    "him",
    "her",
    "them",
    "us",
    "this",
    "that",
    "these",
    "those",
    "with",
    "from",
    "up",
    "down",
    "into",
    "over",
    "under",
    "again",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "just",
    "now",
    "during",
    "before",
    "after",
    "above",
    "below",
    "off",
    "further",
    # 中文停用词
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "人",
    "都",
    "一",
    "一个",
    "上",
    "也",
    "很",
    "到",
    "说",
    "要",
    "去",
    "你",
    "会",
    "着",
    "没有",
    "看",
    "好",
    "自己",
    "这",
    "那",
    "么",
    "吧",
    "呢",
    "啊",
    "呀",
    "哦",
    "嗯",
    "哪",
    "什么",
    "怎么",
    "为什么",
    "如何",
    "能",
    "可以",
    "应该",
    "必须",
    "已经",
    "还是",
    "或者",
    "但是",
    "然后",
    "所以",
    "因为",
    "如果",
    "虽然",
    # 常见的无意义短词
    "er",
    "ly",
    "ed",
    "al",
    "ic",
    "re",
    "un",
    "pre",
    "de",
    "ex",
    "sub",
}

# 最小词长度阈值
MIN_WORD_LENGTH = 2


class TrieNode:
    """A single node in the Trie data structure."""

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end_of_word: bool = False
        self.frequency: int = 0  # Track how often this word is used
        self.word: Optional[str] = None  # Store the complete word at end nodes
        self.original_word: Optional[str] = None  # Store the original complete query


class AutocompleteTrie:
    """
    Trie data structure optimized for autocomplete functionality.
    Features:
    - Persistence to disk
    - Frequency-based ranking
    - LRU cache for fast lookups
    - Prefix-based suggestions
    """

    def __init__(
        self, persistence_file: str = "models/autocomplete/enhanced_trie_data.pkl"
    ):
        self.root = TrieNode()
        self.persistence_file = persistence_file
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.enable_word_segmentation = True  # 启用分词功能
        self._create_data_dir()
        self.load_from_disk()

    def _create_data_dir(self):
        """Create data directory if it doesn't exist."""
        os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)

    def _is_valid_suggestion(self, word: str, original_word: str = None) -> bool:
        """
        检查建议是否有效（过滤停用词和过短的词）

        Args:
            word (str): 要检查的词
            original_word (str): 原始完整词（如果有的话）

        Returns:
            bool: 是否为有效建议
        """
        # 使用原始词进行检查，如果没有则使用当前词
        check_word = (original_word or word).strip().lower()

        # 过滤过短的词
        if len(check_word) < MIN_WORD_LENGTH:
            return False

        # 过滤停用词
        if check_word in STOP_WORDS:
            return False

        # 过滤只包含停用词的组合
        words = re.findall(r"\b\w+\b", check_word.lower())
        if words and all(w in STOP_WORDS for w in words):
            return False

        return True

    def _segment_text(self, text: str) -> List[str]:
        """
        对文本进行分词处理，生成多个搜索索引

        Args:
            text (str): 输入文本

        Returns:
            List[str]: 分词后的所有可能子串
        """
        segments = []
        text = text.strip()

        # 添加完整查询
        segments.append(text)

        # 中文分词
        chinese_words = jieba.lcut(text)
        chinese_words = [
            w.strip()
            for w in chinese_words
            if w.strip()
            and len(w.strip()) > 1
            and w.strip() not in '，。？！；：""（）【】'
        ]

        # 生成中文词组合
        for i in range(len(chinese_words)):
            for j in range(i + 1, len(chinese_words) + 1):
                segment = "".join(chinese_words[i:j]).strip()
                if len(segment) > 1:
                    segments.append(segment)

        # 英文分词
        english_words = re.findall(r"\b[a-zA-Z]+\b", text)
        english_words = [w.strip() for w in english_words if len(w.strip()) > 1]

        # 生成英文词组合
        for i in range(len(english_words)):
            for j in range(i + 1, len(english_words) + 1):
                segment = " ".join(english_words[i:j]).strip()
                if len(segment) > 1:
                    segments.append(segment)

        # 单独的中文词
        for word in chinese_words:
            if len(word) > 1:
                segments.append(word)
                # 对长词再次分解（处理 jieba 可能的过度组合）
                if len(word) > 2:
                    # 尝试提取2-3字的子词
                    for i in range(len(word) - 1):
                        for j in range(i + 2, min(i + 4, len(word) + 1)):
                            sub_word = word[i:j]
                            if len(sub_word) >= 2:
                                segments.append(sub_word)

        # 单独的英文单词
        for word in english_words:
            if len(word) > 1:
                segments.append(word)

        # 去重并返回
        return list(set(segments))

    def insert(self, word: str, frequency: int = 1):
        """
        Insert a word into the trie with optional frequency.
        支持分词处理，将长查询拆分成多个可搜索的片段。

        Args:
            word (str): The word to insert
            frequency (int): The frequency/weight of this word
        """
        if not word or not word.strip():
            return

        original_word = word.strip()

        if self.enable_word_segmentation:
            segments = self._segment_text(original_word)
            for segment in segments:
                self._insert_single(segment, original_word, frequency)
        else:
            self._insert_single(original_word, original_word, frequency)

    def _insert_single(self, key: str, original_word: str, frequency: int):
        """插入单个词到 Trie"""
        key = key.lower()
        node = self.root

        # Traverse the trie, creating nodes as needed
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        # Mark end of word and update frequency
        if node.is_end_of_word:
            node.frequency += frequency
        else:
            node.is_end_of_word = True
            node.frequency = frequency
            node.word = key
            node.original_word = original_word

        self.word_frequencies[original_word] += frequency

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
        现在支持智能匹配，包括分词匹配和模糊匹配。

        Args:
            prefix (str): The prefix to search for
            max_suggestions (int): Maximum number of suggestions to return

        Returns:
            List[Dict]: List of suggestions with word and frequency
        """
        if not prefix or not prefix.strip():
            return []

        return self.smart_search(prefix, max_suggestions)

    def smart_search(
        self, query: str, max_suggestions: int = 10
    ) -> List[Dict[str, any]]:
        """
        智能搜索，结合多种策略，并过滤停用词和过短结果

        Args:
            query (str): 搜索查询
            max_suggestions (int): 最大建议数量

        Returns:
            List[Dict]: 搜索建议列表
        """
        if not query or not query.strip():
            return []

        query = query.strip()

        # 如果查询本身就是停用词或过短，直接返回空结果
        if not self._is_valid_suggestion(query):
            return []

        all_suggestions = []

        # 1. 精确前缀匹配
        exact_matches = self._prefix_search(query.lower(), max_suggestions)
        # 过滤有效建议
        filtered_exact = [
            {**s, "match_type": "exact"}
            for s in exact_matches
            if self._is_valid_suggestion(s["word"], s.get("original_word"))
        ]
        all_suggestions.extend(filtered_exact)

        # 2. 分词后的部分匹配
        if self.enable_word_segmentation:
            segments = self._segment_text(query)
            for segment in segments:
                if segment.lower() != query.lower():
                    partial_matches = self._prefix_search(
                        segment.lower(), max_suggestions // 2
                    )
                    # 过滤有效建议
                    filtered_partial = [
                        {**s, "match_type": "partial"}
                        for s in partial_matches
                        if self._is_valid_suggestion(s["word"], s.get("original_word"))
                    ]
                    all_suggestions.extend(filtered_partial)

        # 3. 模糊匹配（编辑距离 <= 1）
        if len(query) > 2:  # 只对长度大于2的查询进行模糊匹配
            fuzzy_matches = self._fuzzy_search(
                query.lower(), max_distance=1, max_suggestions=max_suggestions // 2
            )
            # 过滤有效建议
            filtered_fuzzy = [
                {**s, "match_type": "fuzzy"}
                for s in fuzzy_matches
                if self._is_valid_suggestion(s["word"], s.get("original_word"))
            ]
            all_suggestions.extend(filtered_fuzzy)

        # 去重并排序
        seen = set()
        unique_suggestions = []
        for s in all_suggestions:
            # 使用原始词作为去重键
            original = s.get("original_word", s["word"])
            if original not in seen:
                seen.add(original)
                unique_suggestions.append(s)

        # 按匹配类型和频率排序
        match_type_priority = {"exact": 0, "partial": 1, "fuzzy": 2}
        unique_suggestions.sort(
            key=lambda x: (match_type_priority.get(x["match_type"], 3), -x["frequency"])
        )

        return unique_suggestions[:max_suggestions]

    def _prefix_search(
        self, prefix: str, max_suggestions: int = 10
    ) -> List[Dict[str, any]]:
        """
        传统的前缀搜索

        Args:
            prefix (str): 搜索前缀
            max_suggestions (int): 最大建议数量

        Returns:
            List[Dict]: 搜索结果
        """
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

    def _fuzzy_search(
        self, prefix: str, max_distance: int = 1, max_suggestions: int = 10
    ) -> List[Dict[str, any]]:
        """
        模糊搜索，支持编辑距离

        Args:
            prefix (str): 搜索前缀
            max_distance (int): 最大编辑距离
            max_suggestions (int): 最大建议数量

        Returns:
            List[Dict]: 模糊匹配结果
        """
        suggestions = []

        def dfs(node, current_word, distance, pos):
            if distance > max_distance:
                return

            if node.is_end_of_word and len(current_word) >= len(prefix) - max_distance:
                suggestions.append(
                    {
                        "word": current_word,
                        "original_word": getattr(node, "original_word", current_word),
                        "frequency": node.frequency,
                        "distance": distance,
                    }
                )
                if len(suggestions) >= max_suggestions * 3:  # 收集更多候选
                    return

            # 如果已经处理完整个前缀，继续遍历子节点
            if pos >= len(prefix):
                for char, child in node.children.items():
                    dfs(child, current_word + char, distance, pos + 1)
                return

            target_char = prefix[pos].lower()

            for char, child in node.children.items():
                if char == target_char:
                    # 精确匹配
                    dfs(child, current_word + char, distance, pos + 1)
                elif distance < max_distance:
                    # 替换
                    dfs(child, current_word + char, distance + 1, pos + 1)
                    # 插入
                    dfs(child, current_word + char, distance + 1, pos)

            # 删除（跳过当前前缀字符）
            if distance < max_distance and pos < len(prefix):
                dfs(node, current_word, distance + 1, pos + 1)

        dfs(self.root, "", 0, 0)

        # 按距离和频率排序
        suggestions.sort(key=lambda x: (x["distance"], -x["frequency"]))
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
            original = getattr(node, "original_word", current_word)
            suggestions.append(
                {
                    "word": current_word,
                    "original_word": original,
                    "frequency": node.frequency,
                }
            )

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
                "original_word": getattr(node, "original_word", None),
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
            node.original_word = node_data.get("original_word")

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
