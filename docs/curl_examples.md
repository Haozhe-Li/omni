# Autocomplete API - Curl Examples

## 基础使用示例

### 1. 健康检查

```bash
curl -X GET "http://localhost:8000/health"
```

### 2. 加载搜索查询数据

```bash
curl -X POST "http://localhost:8000/autocomplete/load" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/Users/lihaozhe/Coding/omni/data/search_queries.txt"
  }'
```

### 3. 获取自动补全建议

```bash
# 搜索以 "python" 开头的查询
curl -X POST "http://localhost:8000/autocomplete/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "python",
    "max_suggestions": 5
  }'

# 搜索以 "machine" 开头的查询
curl -X POST "http://localhost:8000/autocomplete/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "machine",
    "max_suggestions": 10
  }'

# 搜索以 "web" 开头的查询
curl -X POST "http://localhost:8000/autocomplete/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "web",
    "max_suggestions": 8
  }'
```

### 4. 更新词频（当用户选择某个建议时）

```bash
curl -X POST "http://localhost:8000/autocomplete/update" \
  -H "Content-Type: application/json" \
  -d '{
    "word": "python programming",
    "increment": 3
  }'
```

### 5. 获取 Trie 统计信息

```bash
curl -X GET "http://localhost:8000/autocomplete/stats"
```

### 6. 手动保存 Trie 到磁盘

```bash
curl -X POST "http://localhost:8000/autocomplete/save" \
  -H "Content-Type: application/json"
```

### 7. 清除 LRU 缓存

```bash
curl -X POST "http://localhost:8000/autocomplete/clear-cache" \
  -H "Content-Type: application/json"
```

## 高级使用示例

### 测试中文查询支持

```bash
# 首先加载包含中文的数据文件
curl -X POST "http://localhost:8000/autocomplete/load" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/chinese_queries.txt"
  }'

# 搜索中文前缀
curl -X POST "http://localhost:8000/autocomplete/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "机器学习",
    "max_suggestions": 5
  }'
```

### 批量测试不同前缀

```bash
# 测试多个不同长度的前缀
for prefix in "p" "py" "pyt" "pyth" "pytho" "python"; do
  echo "Testing prefix: $prefix"
  curl -X POST "http://localhost:8000/autocomplete/suggest" \
    -H "Content-Type: application/json" \
    -d "{
      \"prefix\": \"$prefix\",
      \"max_suggestions\": 3
    }" | jq '.suggestions[].word'
  echo ""
done
```

### 性能测试

```bash
# 测试大量请求的性能
echo "Performance testing..."
time for i in {1..100}; do
  curl -s -X POST "http://localhost:8000/autocomplete/suggest" \
    -H "Content-Type: application/json" \
    -d '{
      "prefix": "web",
      "max_suggestions": 5
    }' > /dev/null
done
```

## 错误测试

### 测试无效文件路径

```bash
curl -X POST "http://localhost:8000/autocomplete/load" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/nonexistent/file.txt"
  }'
```

### 测试空前缀

```bash
curl -X POST "http://localhost:8000/autocomplete/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "",
    "max_suggestions": 5
  }'
```

### 测试不存在的前缀

```bash
curl -X POST "http://localhost:8000/autocomplete/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "xyzzzzabc123",
    "max_suggestions": 5
  }'
```

## 响应格式示例

### 成功的建议响应

```json
{
  "suggestions": [
    {
      "word": "python programming",
      "frequency": 3
    },
    {
      "word": "python web scraping",
      "frequency": 1
    }
  ],
  "prefix": "python",
  "count": 2
}
```

### 统计信息响应

```json
{
  "stats": {
    "total_words": 1000,
    "total_frequency": 1250,
    "persistence_file": "data/trie_data.pkl"
  },
  "top_queries": [
    {
      "word": "machine learning",
      "frequency": 5
    }
  ]
}
```

## 启动服务器

在运行这些 curl 命令之前，请确保 FastAPI 服务器正在运行：

```bash
cd /Users/lihaozhe/Coding/omni
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

然后你可以在浏览器中访问：

- API 文档: http://localhost:8000/docs
- 替代文档: http://localhost:8000/redoc
