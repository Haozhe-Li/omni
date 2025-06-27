from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models
from langgraph.graph import StateGraph, START, MessagesState, END
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from core.sources import ss


def web_search(query: str) -> str:
    """Perform a web search using Google Serper API."""
    search = GoogleSerperAPIWrapper(k=3)
    results = search.results(query)
    return results.get("organic", [])


def quick_search(query: str) -> str:
    """Perform a quick search and return a summary of the results.

    Args:
        query (str): The search query to perform.

    Returns:
        str: A summary of the search results.
    """
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
    context = "\n\n".join(result["snippet"] for result in search_results)
    return context


light_agent = create_react_agent(
    model=default_llm_models.light_agent_model,
    tools=[quick_search],
    prompt=(
        "Your name is Omni, and to be more specific you are light agent in Omni compound system\n\n"
        "### Instructions\n"
        "- you will be given a question, and you need to answer it as best as you can\n"
        "- when answering questions, use markdown to format your answer.\n"
        "- if you need to search the web, use the `quick_search` tool to get the internet search results.\n\n"
        "### About Yourself\n"
        "- You are a light agent in Omni compound system, you are responsible for answering quick questions and get instant results.\n"
        "- You have the access to internet, so you can search the web for information.\n"
        "- However, if users ask you to do something that requires more complex reasoning or planning, you should refer them to the Omni mode.\n"
        "Now, do your best to answer the question.\n\n"
    ),
    name="light_agent",
)

light = (
    StateGraph(MessagesState)
    .add_node(light_agent, destinations=(END,))
    .add_edge(START, "light_agent")
    .add_edge("light_agent", END)
    .compile()
)
