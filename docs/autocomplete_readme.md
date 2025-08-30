# Autocomplete Trie 功能

这个模块为 Omni 项目添加了基于 Trie 数据结构的自动补全功能。

## 功能特性

- **Trie 数据结构**: 高效的前缀搜索和自动补全
- **持久化存储**: 数据保存到磁盘，支持重启后恢复
- **LRU 缓存**: 使用 Python 的`@lru_cache`装饰器优化查询性能
- **频率统计**: 基于使用频率排序建议结果
- **RESTful API**: FastAPI 接口供前端调用

## API 接口

### 1. 获取自动补全建议

```http
POST /autocomplete/suggest
Content-Type: application/json

{
    "prefix": "python",
    "max_suggestions": 10
}
```

**响应:**

```json
{
  "suggestions": [
    { "word": "python programming", "frequency": 5 },
    { "word": "python tutorial", "frequency": 3 },
    { "word": "python web scraping", "frequency": 2 }
  ],
  "prefix": "python",
  "count": 3
}
```

### 2. 更新词频

```http
POST /autocomplete/update
Content-Type: application/json

{
    "word": "python programming",
    "increment": 1
}
```

**响应:**

```json
{
  "message": "Updated frequency for 'python programming' by 1",
  "word": "python programming",
  "increment": 1
}
```

### 3. 加载数据文件

```http
POST /autocomplete/load
Content-Type: application/json

{
    "file_path": "data/search_queries.txt"
}
```

**响应:**

```json
{
  "message": "Successfully loaded data from data/search_queries.txt",
  "stats": {
    "total_words": 1000,
    "total_frequency": 1000,
    "persistence_file": "data/trie_data.pkl"
  }
}
```

### 4. 获取统计信息

```http
GET /autocomplete/stats
```

**响应:**

```json
{
  "stats": {
    "total_words": 1000,
    "total_frequency": 1500,
    "persistence_file": "data/trie_data.pkl"
  },
  "top_queries": [
    { "word": "python programming", "frequency": 10 },
    { "word": "machine learning", "frequency": 8 }
  ]
}
```

### 5. 手动保存

```http
POST /autocomplete/save
```

### 6. 清除缓存

```http
POST /autocomplete/clear-cache
```

## 使用方法

### 1. 初始化 Trie 数据

首先运行初始化脚本来加载搜索查询数据：

```bash
cd /Users/lihaozhe/Coding/omni
python scripts/init_trie.py
```

### 2. 启动服务

```bash
python main.py
```

或者如果你使用 uvicorn：

```bash
uvicorn main:app --reload
```

### 3. 测试功能

运行测试脚本验证功能：

```bash
python tests/test_trie.py
```

### 4. 前端集成示例

```javascript
// 获取自动补全建议
async function getAutocompleteSuggestions(prefix) {
  const response = await fetch("http://localhost:8000/autocomplete/suggest", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prefix: prefix,
      max_suggestions: 10,
    }),
  });

  const data = await response.json();
  return data.suggestions;
}

// 更新选中建议的频率
async function updateSuggestionFrequency(selectedWord) {
  await fetch("http://localhost:8000/autocomplete/update", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      word: selectedWord,
      increment: 1,
    }),
  });
}

// 使用示例
const suggestions = await getAutocompleteSuggestions("python");
console.log(suggestions);
// 当用户选择某个建议时
await updateSuggestionFrequency("python programming");
```

## 数据格式

搜索查询文件 (`data/search_queries.txt`) 应该是纯文本格式，每行一个查询：

```
machine learning tutorial
python programming
web development
data science
...
```

## 性能优化

- **LRU 缓存**: `search`方法缓存 1000 个结果，`get_suggestions`方法缓存 500 个结果
- **持久化**: 数据自动保存到 `data/trie_data.pkl`
- **频率排序**: 建议按使用频率排序，常用查询排在前面
- **批量插入**: 支持批量加载大量数据

## 文件结构

```
core/
├── trie.py                    # Trie数据结构实现
scripts/
├── init_trie.py              # 初始化脚本
tests/
├── test_trie.py              # 测试脚本
data/
├── search_queries.txt        # 搜索查询数据
├── trie_data.pkl            # 持久化的Trie数据
```

## 扩展建议

1. **多语言支持**: 添加对中文、日文等语言的支持
2. **模糊匹配**: 支持拼写错误的模糊匹配
3. **分词支持**: 对长查询进行分词处理
4. **同义词支持**: 添加同义词建议
5. **上下文感知**: 根据用户历史查询提供个性化建议
