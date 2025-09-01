# 🤖 Omni - Multi-Agent AI Assistant System

<div align="center">
  <img src="./docs/assets/omni_cover.gif" alt="Omni Cover" width="60%" />
  
  <p align="center">
    <strong>A powerful multi-agent AI system that combines specialized agents for comprehensive task handling</strong>
  </p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

English | [简体中文](README_ZH.md)

</div>

## 🌟 Features

Omni is an intelligent multi-agent system that orchestrates specialized AI agents to handle diverse tasks:

### 🔬 **Research Agent**

- Advanced web search and information retrieval
- Content summarization and analysis
- Real-time data collection with caching

### 💻 **Coding Agent**

- Code generation and debugging
- Multi-language programming support
- Code review and optimization

### 🧮 **Math Agent**

- Complex mathematical problem solving
- Statistical analysis and calculations
- Formula derivation and explanation

### 🌐 **Web Browsing Agent**

- Intelligent web scraping
- Content extraction and processing
- URL analysis and summarization

### ☁️ **Weather Agent**

- Real-time weather information
- Location-based forecasts
- Weather data analysis

### 📝 **Summarizing Agent**

- Document and content summarization
- Key insights extraction
- Multi-format content processing

### 🎯 **Smart Routing**

- Intelligent agent selection based on query type
- Seamless handoff between agents
- Optimized response generation

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Haozhe-Li/omni.git
   cd omni
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**

   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API**
   - API Documentation: `http://localhost:8000/docs`
   - Interactive API: `http://localhost:8000/redoc`

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t omni .

# Run the container
docker run -p 8080:8080 --env-file .env omni
```

## 📖 API Usage

### Stream Chat Endpoint

Send queries to the multi-agent system:

```bash
curl -X POST "http://localhost:8000/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather like in Tokyo?"}],
    "mode": "supervisor",
    "location": "Tokyo, Japan",
    "useCache": true
  }'
```

### Autocomplete Suggestions

Get intelligent autocomplete suggestions:

```bash
curl -X POST "http://localhost:8000/autocomplete" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "what is the weather",
    "max_suggestions": 5
  }'
```

### Query Suggestions

Get related query suggestions:

```bash
curl -X POST "http://localhost:8000/suggestion" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "machine learning algorithms"
  }'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│   Supervisor     │────│  Agent Router   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────────┐
                    │        Specialized Agents        │
                    ├──────────────────────────────────┤
                    │  Research │ Coding │ Math │ ...  │
                    └──────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────────┐
                    │     External Services & Cache    │
                    │  Vector DB │ Search │ Weather   │
                    └──────────────────────────────────┘
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key

# Search Configuration
SERPER_API_KEY=your_serper_api_key

# Weather Configuration
OPENWEATHERMAP_API_KEY=your_weather_api_key

# Cache Configuration
INGEST_CACHE=true
```

### Agent Modes

- **`supervisor`**: Full multi-agent system with intelligent routing
- **`light`**: Lightweight single-agent mode for faster responses

## 📊 Performance Features

- **Semantic Caching**: Intelligent response caching for faster queries
- **Vector Database**: Efficient similarity search and retrieval
- **Streaming Responses**: Real-time response streaming
- **Autocomplete**: Enhanced trie-based autocomplete system
- **Multi-language Support**: Automatic language detection and response matching

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_trie.py

# Interactive testing with Jupyter notebooks
jupyter lab tests/
```

## 📁 Project Structure

```
omni/
├── core/                    # Core system components
│   ├── agents/             # Specialized agent implementations
│   ├── embedding.py        # Text embedding utilities
│   ├── supervisors.py      # Agent orchestration
│   ├── vectordb.py         # Vector database interface
│   └── utils.py            # Utility functions
├── models/                 # Pre-trained models and data
├── tests/                  # Test files and notebooks
├── docs/                   # Documentation and assets
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
└── Dockerfile             # Container configuration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the agent framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Qdrant](https://qdrant.tech/) for vector database capabilities

## 📞 Support

If you have any questions or need help, please:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/Haozhe-Li/omni/issues)
3. Create a new issue if needed

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/Haozhe-Li">Haozhe Li</a></sub>
</div>
