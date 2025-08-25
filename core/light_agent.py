from typing import Optional
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from core.llm_models import default_llm_models
from core.sources import ss
from core.semantic_search_cache import semantic_cache


def web_search(querys: list[str]) -> Optional[tuple[list[dict], str, dict]]:
    """Perform a web search using Google Serper API.

    Args:
        querys (list[str]): A list of search queries.

    Returns:
        Optional[tuple[list[dict], str, dict]]: A tuple containing the search results, answer box, and knowledge graph.
    """
    print(f"Performing web search for queries: {querys}")
    search = GoogleSerperAPIWrapper(k=3)
    results = []
    for query in querys:
        result = search.results(query)
        results.extend(result.get("organic", []))
    answer_box = result.get("answerBox", "")
    knowledge_graph = result.get("knowledgeGraph", {})
    return results, answer_box, knowledge_graph


def quick_search(query: str) -> str:
    """Perform a quick search and return a summary of the results.

    Args:
        query (str): The search query to perform.

    Returns:
        str: A summary of the search results.
    """
    cached_sources = semantic_cache.get(query, threshold=0.7)
    if cached_sources:
        print(f"Using cached sources for query: {query}")
        ss.set_sources(cached_sources)
        context = "\n\n".join(
            f"{source['title']}: {source['snippet']} ({source['url']})"
            for source in cached_sources
        )
        return context
    search_results, answer_box, knowledge_graph = web_search([query])
    if not search_results:
        return "No search results found."
    urls = [result["link"] for result in search_results]
    # assign sources variable, sources is a list of key: value pairs
    sources = [
        {
            "query": query,
            "url": url,
            "title": result["title"],
            "snippet": result["snippet"],
            "aviod_cache": False,
        }
        for url, result in zip(urls, search_results)
    ]
    context = "\n\n".join(result["snippet"] for result in search_results)
    if answer_box:
        context = (
            f"Answer from Google, refer this answer directly: {answer_box.get('answer', '')}\n\n"
            + context
        )
        sources.append(
            {
                "query": query,
                "url": answer_box.get("sourceLink", "N/A"),
                "title": answer_box.get("title", "N/A"),
                "snippet": answer_box.get("answer", ""),
                "aviod_cache": False,
            }
        )
    if knowledge_graph:
        context = (
            f"Knowledge Graph: {knowledge_graph.get('description', '')}\n\n" + context
        )
        sources.append(
            {
                "query": query,
                "url": knowledge_graph.get("descriptionLink", "N/A"),
                "title": knowledge_graph.get("Apple", "N/A"),
                "snippet": knowledge_graph.get("description", ""),
                "aviod_cache": False,
            }
        )
    ss.set_sources(sources)
    return context


light_agent = create_react_agent(
    model=default_llm_models.light_agent_model,
    tools=[quick_search],
    prompt=(
        "You are **Omni Light**, a specialized AI agent for quick, accurate answers.\n\n"
        "## Core Rules\n"
        "1. **Think First**: Can you answer with existing knowledge? If yes, answer directly without searching.\n"
        "2. **Search Sparingly**: Only use `quick_search` for current events, real-time data, or recent information you don't know.\n"
        "3. **One Search Max**: You can only search once per conversation.\n"
        "4. **Clear Format**: Use Markdown for responses.\n\n"
        "## Search Only For:\n"
        "- Current news/events\n"
        "- Real-time data (stocks, weather, scores)\n"
        "- Recent company/product updates\n\n"
        "## Answer Directly For:\n"
        "- General knowledge\n"
        "- Programming help\n"
        "- Math/science explanations\n"
        "- Historical facts\n"
        "- How-to guides\n\n"
        "For complex multi-step tasks, recommend **Omni Compound Mode**.\n\n"
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
