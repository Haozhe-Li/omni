from langchain_community.utilities import GoogleSerperAPIWrapper
import nest_asyncio
from langgraph.prebuilt import create_react_agent
from core.globalvaris import GROQ_CHAT_MODEL_FAST

model = f"groq:{GROQ_CHAT_MODEL_FAST}"

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
    search = GoogleSerperAPIWrapper(k=3)
    results = search.results(query)
    return results.get("organic", [])


def research(query: str) -> str:
    """Perform a research task by loading web pages and performing a web search."""
    search_results = web_search(query)
    if not search_results:
        return "No search results found."
    urls = [result["link"] for result in search_results]
    # assign sources variable, sources is a list of key: value pairs
    sources = [
        {"url": url, "title": result["title"]}
        for url, result in zip(urls, search_results)
    ]
    ss.set_sources(sources)
    web_content = load_web_page(urls)
    return web_content if web_content else "No content found on the provided URLs."


research_agent = create_react_agent(
    model=model,
    tools=[research],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do anything else.\n"
        "- If you are provided with a task similar to previous one, you must dive deeper, which means change your query to be more specific.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- Respond your answer in <agent_response> tag\n"
    ),
    name="research_agent",
)
