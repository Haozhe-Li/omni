#!/usr/bin/env python3
"""
测试现有 Trie 数据的停用词过滤功能
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_existing_data():
    """测试现有数据的过滤功能"""
    print("=== 测试现有数据的停用词过滤功能 ===")

    # 使用现有的 Trie 实例
    from core.trie import autocomplete_trie as trie

    print(f"Trie 统计信息: {trie.get_stats()}")

    # 测试停用词
    print("\n1. 测试停用词查询:")
    stop_word_queries = [
        "in",
        "and",
        "the",
        "a",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
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
        "with",
        "from",
        "up",
        "down",
        "into",
        "over",
        "under",
        "again",
        "then",
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
    ]

    filtered_count = 0
    for query in stop_word_queries:
        results = trie.smart_search(query, max_suggestions=5)
        if len(results) == 0:
            filtered_count += 1
        else:
            print(f"警告: 停用词 '{query}' 仍然返回了 {len(results)} 个结果")

    print(f"成功过滤了 {filtered_count}/{len(stop_word_queries)} 个停用词查询")

    # 测试过短查询
    print("\n2. 测试过短查询:")
    short_queries = ["a", "I", "s", "t", "之", "我"]

    short_filtered = 0
    for query in short_queries:
        results = trie.smart_search(query, max_suggestions=5)
        if len(results) == 0:
            short_filtered += 1
        else:
            print(f"警告: 过短查询 '{query}' 仍然返回了 {len(results)} 个结果")

    print(f"成功过滤了 {short_filtered}/{len(short_queries)} 个过短查询")

    # 测试一些正常查询仍然有效
    print("\n3. 测试正常查询:")
    normal_queries = [
        "machine",
        "learn",
        "python",
        "data",
        "algorithm",
        "neural",
        "deep",
    ]

    working_count = 0
    for query in normal_queries:
        results = trie.smart_search(query, max_suggestions=3)
        if len(results) > 0:
            working_count += 1
            print(f"'{query}': {len(results)} 个结果")
            for result in results[:2]:  # 只显示前2个
                print(f"  - {result.get('original_word', result['word'])}")
        else:
            print(f"警告: 正常查询 '{query}' 没有返回结果")

    print(f"{working_count}/{len(normal_queries)} 个正常查询工作正常")

    print("\n测试完成!")


if __name__ == "__main__":
    test_existing_data()
