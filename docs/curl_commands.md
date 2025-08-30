# 单独的 Curl 命令示例 - 复制即用

## 前提条件

# 确保 FastAPI 服务器正在运行:

# uvicorn main:app --reload --host 0.0.0.0 --port 8000

## 1. 检查服务器状态

curl -X GET "http://localhost:8000/health"

## 2. 加载搜索查询数据到 Trie

curl -X POST "http://localhost:8000/autocomplete/load" \
 -H "Content-Type: application/json" \
 -d '{
"file_path": "/Users/lihaozhe/Coding/omni/data/search_queries.txt"
}'

## 3. 测试自动补全 - Python 相关

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "python",
"max_suggestions": 8
}'

## 4. 测试自动补全 - Machine Learning 相关

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "machine",
"max_suggestions": 5
}'

## 5. 测试自动补全 - Web 开发相关

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "web",
"max_suggestions": 6
}'

## 6. 测试自动补全 - Data 相关

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "data",
"max_suggestions": 7
}'

## 7. 更新词频（模拟用户选择了某个建议）

curl -X POST "http://localhost:8000/autocomplete/update" \
 -H "Content-Type: application/json" \
 -d '{
"word": "machine learning tutorial",
"increment": 5
}'

## 8. 测试短前缀

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "ai",
"max_suggestions": 10
}'

## 9. 获取 Trie 统计和热门查询

curl -X GET "http://localhost:8000/autocomplete/stats"

## 10. 手动保存 Trie 数据

curl -X POST "http://localhost:8000/autocomplete/save" \
 -H "Content-Type: application/json"

## 11. 清除 LRU 缓存

curl -X POST "http://localhost:8000/autocomplete/clear-cache" \
 -H "Content-Type: application/json"

## 12. 测试边界情况 - 空前缀

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "",
"max_suggestions": 5
}'

## 13. 测试边界情况 - 不存在的前缀

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "qqqqqqzzzzzz",
"max_suggestions": 5
}'

## 使用 jq 美化 JSON 输出 (需要先安装 jq: brew install jq)

# 例如:

curl -X POST "http://localhost:8000/autocomplete/suggest" \
 -H "Content-Type: application/json" \
 -d '{
"prefix": "python",
"max_suggestions": 5
}' | jq '.'

## 一行测试命令 - 测试完整流程

curl -X POST "http://localhost:8000/autocomplete/load" -H "Content-Type: application/json" -d '{"file_path": "/Users/lihaozhe/Coding/omni/data/search_queries.txt"}' && curl -X POST "http://localhost:8000/autocomplete/suggest" -H "Content-Type: application/json" -d '{"prefix": "python", "max_suggestions": 5}' | jq '.'
