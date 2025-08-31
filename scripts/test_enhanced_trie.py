#!/usr/bin/env python3
"""
测试增强版 Trie 的智能搜索功能
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trie import AutocompleteTrie


def test_enhanced_trie():
    """测试增强版 Trie 功能"""
    print("🧪 测试增强版 Trie 智能搜索功能")
    print("=" * 50)

    # 加载增强版 Trie
    trie = AutocompleteTrie("models/autocomplete/enhanced_trie_data.pkl")

    # 详细测试用例
    test_cases = [
        {
            "query": "天气",
            "expected": "应该匹配包含天气的查询",
            "description": "中文部分词匹配测试",
        },
        {
            "query": "python",
            "expected": "应该匹配 Python 相关查询",
            "description": "英文部分词匹配测试",
        },
        {
            "query": "餐厅",
            "expected": "应该匹配餐厅相关查询",
            "description": "中文精确匹配测试",
        },
        {
            "query": "邮件",
            "expected": "应该匹配邮件相关查询",
            "description": "中文单词匹配测试",
        },
        {
            "query": "pythn",  # 故意拼错
            "expected": "应该模糊匹配到 python",
            "description": "模糊匹配测试",
        },
        {
            "query": "wether",  # 故意拼错
            "expected": "应该模糊匹配到 weather",
            "description": "英文模糊匹配测试",
        },
        {
            "query": "学习",
            "expected": "应该匹配学习相关查询",
            "description": "中文学习场景测试",
        },
        {
            "query": "schedule",
            "expected": "应该匹配日程相关查询",
            "description": "英文日程场景测试",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        description = test_case["description"]
        expected = test_case["expected"]

        print(f"\n{i}. {description}")
        print(f"   查询: '{query}'")
        print(f"   期望: {expected}")

        # 执行智能搜索
        suggestions = trie.smart_search(query, 5)

        if suggestions:
            print(f"   结果: 找到 {len(suggestions)} 个匹配")
            for j, suggestion in enumerate(suggestions, 1):
                original = suggestion.get("original_word", suggestion["word"])
                match_type = suggestion.get("match_type", "unknown")
                frequency = suggestion["frequency"]
                print(f"     {j}. {original}")
                print(f"        (频率: {frequency}, 匹配类型: {match_type})")
        else:
            print("   结果: ❌ 无匹配结果")

    print("\n" + "=" * 50)
    print("📈 性能测试...")

    # 性能测试
    import time

    performance_queries = ["天气", "python", "学习", "开发", "weather", "tutorial"]

    start_time = time.time()
    for query in performance_queries * 100:  # 执行600次查询
        trie.smart_search(query, 10)
    end_time = time.time()

    avg_time = (end_time - start_time) / (len(performance_queries) * 100) * 1000
    print(f"平均查询时间: {avg_time:.2f} ms")

    print("\n✅ 增强版 Trie 测试完成！")


if __name__ == "__main__":
    test_enhanced_trie()
