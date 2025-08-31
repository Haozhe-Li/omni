#!/usr/bin/env python3
"""
增强版 Trie 初始化脚本
支持中文分词、模糊匹配和智能搜索
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trie import AutocompleteTrie


def main():
    """主函数"""
    print("🚀 初始化增强版自动补全Trie (支持分词和模糊匹配)")
    print("=" * 60)

    # 创建增强版 Trie
    trie = AutocompleteTrie("models/autocomplete/enhanced_trie_data.pkl")

    # 数据文件路径
    data_file = "data/search_queries.txt"

    print(f"📁 加载数据文件: {os.path.abspath(data_file)}")

    # 加载数据
    trie.load_from_text_file(data_file)

    print("💾 保存增强版Trie到磁盘...")
    trie.save_to_disk()

    # 显示统计信息
    stats = trie.get_stats()
    print("📊 统计信息:")
    print(f"   - 总词汇数: {stats['total_words']}")
    print(f"   - 总频率: {stats['total_frequency']}")
    print(f"   - 存储位置: {stats['persistence_file']}")

    print("\n🔍 测试智能搜索功能...\n")

    # 测试用例
    test_cases = [
        "天气",  # 应该匹配 "你好，今天天气怎么样？"
        "python",  # 应该匹配 Python 相关查询
        "餐厅",  # 应该匹配 "请帮我查一下附近的餐厅"
        "邮件",  # 应该匹配 "帮我写一封英文邮件"
        "weather",  # 应该匹配英文天气查询
        "restaurant",  # 应该匹配餐厅相关查询
    ]

    for test_query in test_cases:
        print(f"搜索 '{test_query}':")
        suggestions = trie.smart_search(test_query, 5)

        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                original = suggestion.get("original_word", suggestion["word"])
                match_type = suggestion.get("match_type", "unknown")
                print(
                    f"  {i}. {original} (频率: {suggestion['frequency']}, 类型: {match_type})"
                )
        else:
            print("  无匹配结果")
        print()

    print("🔥 热门查询 (Top 10):")
    top_queries = trie.get_top_queries(10)
    for i, query in enumerate(top_queries, 1):
        print(f"  {i}. {query['word']} (频率: {query['frequency']})")

    print(f"\n✅ 增强版初始化完成! Trie已保存到: {trie.persistence_file}")

    print("\n💡 现在你可以启动FastAPI服务器并测试智能搜索:")
    print("   uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("\n🌐 测试命令示例:")
    print('   curl -X POST "http://localhost:8000/autocomplete/suggest" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"prefix": "天气", "max_suggestions": 10}\'')


if __name__ == "__main__":
    main()
