#!/usr/bin/env python3
"""
测试 Trie 智能搜索的停用词过滤功能
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trie import AutocompleteTrie


def test_stop_words_filtering():
    """测试停用词过滤功能"""
    print("=== 测试 Trie 停用词过滤功能 ===")

    # 创建一个新的 Trie 实例用于测试
    trie = AutocompleteTrie("/tmp/test_trie.pkl")

    # 插入一些测试数据，包括停用词和有效词
    test_queries = [
        "machine learning",
        "deep learning",
        "in",  # 停用词
        "and",  # 停用词
        "ing",  # 停用词
        "out",  # 停用词
        "python programming",
        "the",  # 停用词
        "a",  # 停用词
        "data science",
        "artificial intelligence",
        "how to learn",
        "what is",
        "neural networks",
        "computer vision",
        "是",  # 中文停用词
        "了",  # 中文停用词
        "的",  # 中文停用词
        "机器学习",
        "深度学习",
        "人工智能",
    ]

    # 插入测试数据
    for query in test_queries:
        trie.insert(query, frequency=1)

    # 测试搜索功能
    print("\n1. 测试停用词过滤:")
    test_cases = ["in", "and", "ing", "out", "the", "a", "是", "了", "的"]

    for query in test_cases:
        results = trie.smart_search(query, max_suggestions=10)
        print(f"查询 '{query}': {len(results)} 个结果")
        for result in results:
            print(
                f"  - {result['original_word']} (频率: {result['frequency']}, 类型: {result['match_type']})"
            )

    print("\n2. 测试有效查询:")
    valid_cases = ["machine", "deep", "python", "data", "neural", "机器", "人工"]

    for query in valid_cases:
        results = trie.smart_search(query, max_suggestions=5)
        print(f"查询 '{query}': {len(results)} 个结果")
        for result in results:
            print(
                f"  - {result['original_word']} (频率: {result['frequency']}, 类型: {result['match_type']})"
            )

    print("\n3. 测试过滤器功能 - 通过搜索结果验证:")
    filter_test_cases = [
        ("in", "应该被过滤掉"),
        ("machine", "应该有结果"),
        ("the", "应该被过滤掉"),
        ("deep", "应该有结果"),
        ("的", "应该被过滤掉"),
        ("人工", "应该有结果"),
    ]

    for word, description in filter_test_cases:
        results = trie.smart_search(word, max_suggestions=5)
        print(f"'{word}' ({description}): {len(results)} 个结果")
        if results:
            for result in results[:2]:  # 只显示前2个结果
                print(f"  - {result['original_word']}")
        print()

    # 清理测试文件
    if os.path.exists("/tmp/test_trie.pkl"):
        os.remove("/tmp/test_trie.pkl")

    print("\n测试完成!")


if __name__ == "__main__":
    test_stop_words_filtering()
