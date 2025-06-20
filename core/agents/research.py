from langchain_community.utilities import GoogleSerperAPIWrapper
import nest_asyncio

nest_asyncio.apply()
from langchain_community.document_loaders import WebBaseLoader


def load_web_page(urls: list[str]) -> str:
    """Load a web page and return its content."""
    print("loading urls:", urls)
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
    web_content = load_web_page(urls)
    return web_content if web_content else "No content found on the provided URLs."


from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(
    model="openai:gpt-4.1-nano",
    tools=[research],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)
