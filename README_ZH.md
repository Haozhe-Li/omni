# 🤖 Omni - 多智能体 AI 助手系统

<div align="center">
  <img src="./docs/assets/omni_cover.gif" alt="Omni Cover" width="60%" />
  
  <p align="center">
    <strong>一个强大的多智能体AI系统，结合专业化智能体进行全面任务处理</strong>
  </p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | 简体中文

</div>

## 🌟 功能特性

Omni 是一个智能多智能体系统，通过协调专业化 AI 智能体来处理各种不同的任务：

### 🔬 **研究智能体**

- 高级网络搜索和信息检索
- 内容摘要和分析
- 实时数据收集与缓存

### 💻 **编程智能体**

- 代码生成和调试
- 多语言编程支持
- 代码审查和优化

### 🧮 **数学智能体**

- 复杂数学问题求解
- 统计分析和计算
- 公式推导和解释

### 🌐 **网页浏览智能体**

- 智能网页抓取
- 内容提取和处理
- URL 分析和摘要

### ☁️ **天气智能体**

- 实时天气信息
- 基于位置的天气预报
- 天气数据分析

### 📝 **摘要智能体**

- 文档和内容摘要
- 关键见解提取
- 多格式内容处理

### 🎯 **智能路由**

- 基于查询类型的智能智能体选择
- 智能体间无缝切换
- 优化响应生成

## 🚀 快速开始

### 环境要求

- Python 3.10 或更高版本
- Git

### 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/Haozhe-Li/omni.git
   cd omni
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **设置环境变量**

   ```bash
   cp .env.example .env
   # 编辑 .env 文件，添加你的API密钥
   ```

4. **运行应用程序**

   ```bash
   uvicorn main:app --reload
   ```

5. **访问 API**
   - API 文档: `http://localhost:8000/docs`
   - 交互式 API: `http://localhost:8000/redoc`

## 🐳 Docker 部署

使用 Docker 构建和运行：

```bash
# 构建镜像
docker build -t omni .

# 运行容器
docker run -p 8080:8080 --env-file .env omni
```

## 📖 API 使用

### 流式聊天接口

向多智能体系统发送查询：

```bash
curl -X POST "http://localhost:8000/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "东京的天气怎么样？"}],
    "mode": "supervisor",
    "location": "Tokyo, Japan",
    "useCache": true
  }'
```

### 自动补全建议

获取智能自动补全建议：

```bash
curl -X POST "http://localhost:8000/autocomplete" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "今天天气",
    "max_suggestions": 5
  }'
```

### 查询建议

获取相关查询建议：

```bash
curl -X POST "http://localhost:8000/suggestion" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "机器学习算法"
  }'
```

## 🏗️ 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI 应用  │────│    监督器       │────│  智能体路由器   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────────┐
                    │        专业化智能体              │
                    ├──────────────────────────────────┤
                    │  研究 │ 编程 │ 数学 │ ...      │
                    └──────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────────┐
                    │    外部服务 & 缓存               │
                    │  向量数据库 │ 搜索 │ 天气        │
                    └──────────────────────────────────┘
```

## 🛠️ 配置说明

### 环境变量

创建 `.env` 文件并配置以下变量：

```env
# LLM配置
OPENAI_API_KEY=你的openai_api_key
GROQ_API_KEY=你的groq_api_key

# 搜索配置
SERPER_API_KEY=你的serper_api_key

# 天气配置
OPENWEATHERMAP_API_KEY=你的weather_api_key

# 缓存配置
INGEST_CACHE=true
```

### 智能体模式

- **`supervisor`**: 具有智能路由的完整多智能体系统
- **`light`**: 轻量级单智能体模式，响应更快

## 📊 性能特性

- **语义缓存**: 智能响应缓存，加快查询速度
- **向量数据库**: 高效相似性搜索和检索
- **流式响应**: 实时响应流传输
- **自动补全**: 增强的 Trie 树自动补全系统
- **多语言支持**: 自动语言检测和响应匹配

## 🧪 测试

运行测试套件：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python tests/test_trie.py

# 使用Jupyter notebook进行交互式测试
jupyter lab tests/
```

## 📁 项目结构

```
omni/
├── core/                    # 核心系统组件
│   ├── agents/             # 专业化智能体实现
│   ├── embedding.py        # 文本嵌入工具
│   ├── supervisors.py      # 智能体协调
│   ├── vectordb.py         # 向量数据库接口
│   └── utils.py            # 工具函数
├── models/                 # 预训练模型和数据
├── tests/                  # 测试文件和notebook
├── docs/                   # 文档和资源
├── main.py                 # FastAPI应用程序入口点
├── requirements.txt        # Python依赖
└── Dockerfile             # 容器配置
```

## 🤝 贡献指南

欢迎贡献！请随时提交 Pull Request。

1. Fork 这个仓库
2. 创建你的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://langchain.com/) 提供智能体框架
- [FastAPI](https://fastapi.tiangolo.com/) 提供 Web 框架
- [Qdrant](https://qdrant.tech/) 提供向量数据库能力

## 📞 支持

如果你有任何问题或需要帮助，请：

1. 查看 [文档](docs/)
2. 搜索现有 [issues](https://github.com/Haozhe-Li/omni/issues)
3. 如有需要创建新的 issue

---

<div align="center">
  <sub>用 ❤️ 构建，作者 <a href="https://github.com/Haozhe-Li">Haozhe Li</a></sub>
</div>
