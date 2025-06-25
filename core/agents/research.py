from langchain_community.utilities import GoogleSerperAPIWrapper
import nest_asyncio
from core.llm_models import default_llm_models
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

model = init_chat_model(default_llm_models.research_model)

from langchain_community.document_loaders import WebBaseLoader
from core.sources import ss

nest_asyncio.apply()


def load_web_page(urls: list[str]) -> str:
    """Load a web page and return its content."""
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents


def web_search(query: str) -> str:
    """Perform a web search using Google Serper API."""
    search = GoogleSerperAPIWrapper(k=5)
    results = search.results(query)
    return results.get("organic", [])


@tool(return_direct=True)
def research(query: str) -> str:
    """Perform a research task by loading web pages and performing a web search."""
    search_results = web_search(query)
    if not search_results:
        return "No search results found."
    urls = [result["link"] for result in search_results]
    # assign sources variable, sources is a list of key: value pairs
    sources = [
        {"url": url, "title": result["title"], "snippet": result["snippet"]}
        for url, result in zip(urls, search_results)
    ]
    ss.set_sources(sources)
    # concat all result snippet together as context
    context = "\n\n".join(result["snippet"] for result in search_results)
    print(context)
    return context


research_tool = [research]

# 强制调用名为 "research" 的工具
bound_model = model.bind_tools(
    research_tool,
    tool_choice={
        "type": "function",
        "function": {"name": "research"},
    },
)

research_agent = create_react_agent(
    model=bound_model,
    tools=research_tool,
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do anything else.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- Respond your answer in <agent_response> tag\n"
    ),
    name="research_agent",
)
