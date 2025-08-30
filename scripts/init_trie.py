#!/usr/bin/env python3
"""
初始化和测试自动补全Trie（包含中文支持）
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trie import AutocompleteTrie


def main():
    print("🚀 初始化自动补全Trie (支持中文)")
    print("=" * 50)

    # 创建trie实例
    trie = AutocompleteTrie()

    # 加载搜索查询数据
    data_file = "/Users/lihaozhe/Coding/omni/data/search_queries.txt"
    print(f"📁 加载数据文件: {data_file}")
    trie.load_from_text_file(data_file)

    # 保存到磁盘
    print("💾 保存Trie到磁盘...")
    trie.save_to_disk()

    # 获取统计信息
    stats = trie.get_stats()
    print(f"📊 统计信息:")
    print(f"   - 总词汇数: {stats['total_words']}")
    print(f"   - 总频率: {stats['total_frequency']}")
    print(f"   - 存储位置: {stats['persistence_file']}")

    print("\n🔍 测试英文查询...")
    # 测试英文查询
    test_prefixes_en = ["python", "machine", "web", "data", "ai"]

    for prefix in test_prefixes_en:
        suggestions = trie.get_suggestions(prefix, max_suggestions=5)
        print(f"\n'{prefix}' 的建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['word']} (频率: {suggestion['frequency']})")

    print("\n🔍 测试中文查询...")
    # 测试中文查询
    test_prefixes_zh = ["机器", "人工", "数据", "编程", "网页"]

    for prefix in test_prefixes_zh:
        suggestions = trie.get_suggestions(prefix, max_suggestions=5)
        print(f"\n'{prefix}' 的建议:")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion['word']} (频率: {suggestion['frequency']})")
        else:
            print(f"  没有找到以 '{prefix}' 开头的查询")

    # 测试热门查询
    print("\n🔥 热门查询 (Top 10):")
    top_queries = trie.get_top_queries(limit=10)
    for i, query in enumerate(top_queries, 1):
        print(f"  {i}. {query['word']} (频率: {query['frequency']})")

    # 测试频率更新
    print("\n⬆️ 测试频率更新...")
    test_word = "机器学习教程"
    if trie.search(test_word):
        print(f"更新 '{test_word}' 的频率...")
        trie.update_frequency(test_word, increment=10)
        trie.save_to_disk()

        # 重新测试
        suggestions = trie.get_suggestions("机器", max_suggestions=5)
        print(f"\n更新后 '机器' 的建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['word']} (频率: {suggestion['frequency']})")

    print(f"\n✅ 初始化完成! Trie已保存到: {trie.persistence_file}")
    print("\n💡 现在你可以启动FastAPI服务器并使用curl测试:")
    print("   uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("\n🌐 API文档地址:")
    print("   http://localhost:8000/docs")


if __name__ == "__main__":
    main()
