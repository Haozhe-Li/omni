from langchain_community.utilities import GoogleSerperAPIWrapper
import nest_asyncio
from core.llm_models import default_llm_models
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from core.semantic_search_cache import semantic_cache

model = init_chat_model(default_llm_models.research_model)

from langchain_community.document_loaders import WebBaseLoader
from core.sources import ss

nest_asyncio.apply()


def web_search(
    querys: list[str], k: int = 5, tbs: str = ""
) -> tuple[list[dict], str, dict]:
    print(f"Performing web search for queries: {querys}")
    if not tbs:
        search = GoogleSerperAPIWrapper(k=k)
    else:
        search = GoogleSerperAPIWrapper(k=k, tbs=tbs)
    results = []
    for query in querys:
        result = search.results(query)
        results.extend(result.get("organic", []))
    answer_box = result.get("answerBox", "")
    knowledge_graph = result.get("knowledgeGraph", {})
    return results, answer_box, knowledge_graph


# @tool(return_direct=True)
def research(query: str, time_level: str = "", use_cache: bool = True) -> str:
    """Research a topic using web search and return the context.

    Args:
        query (str): The query to search for.
        time_level (str): The time level for the search. For most of the time you would not use this parameter.
                          Available options are "day", "week", "month", "year".
                          Not passing this parameter will use the default which search results for anytime.
                          You will only use this when the search is time sensitive enough.
                          Defaults to "" means anytime.
        use_cache (bool): Whether to use the semantic search cache.
                          Setting to False will strictly disable the cache and always perform a fresh search.
                          when the time_level is set to "day" or "week", it will always perform a fresh search.
                          Defaults to True, meaning it will use the cache if available.

    Returns:
        str:  A summary of the search results.
    """
    time_level_map = {
        "day": "qdr:d",
        "week": "qdr:w",
        "month": "qdr:m",
        "year": "qdr:y",
    }
    tbs = time_level_map.get(time_level, "")

    if use_cache and time_level not in ["day", "week"]:
        # Use semantic search cache if available
        cached_sources = semantic_cache.get(query, threshold=0.9)
        if cached_sources:
            print(f"Using cached sources for query: {query}")
            ss.set_sources(cached_sources)
            context = "\n\n".join(
                f"{source['title']}: {source['snippet']} ({source['url']})"
                for source in cached_sources
            )
            return context

    search_results, answer_box, knowledge_graph = web_search(
        querys=[query], k=5, tbs=tbs
    )
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
    # concat all result snippet together as context
    context = "\n\n".join(result["snippet"] for result in search_results)
    if answer_box:
        context = (
            f"Answer from Google, refer this answer directly: {answer_box.get('answer')}\n\n"
            + context
        )
        sources.append(
            {
                "url": answer_box.get("sourceLink", "N/A"),
                "title": answer_box.get("title"),
                "snippet": answer_box.get("answer"),
                "aviod_cache": False,
            }
        )
    if knowledge_graph:
        context = (
            f"Knowledge Graph: {knowledge_graph.get('description', '')}\n\n" + context
        )
        sources.append(
            {
                "url": knowledge_graph.get("descriptionLink", "N/A"),
                "title": knowledge_graph.get("Apple", "N/A"),
                "snippet": knowledge_graph.get("description", ""),
                "aviod_cache": False,
            }
        )
    ss.set_sources(sources)
    return context


research_tool = [research]

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
        "You are a professional research agent that searches for information and extracts valuable insights.\n\n"
        "## TOOL PARAMETERS:\n"
        "- **query (str)**: Search query - be specific and clear\n"
        '- **time_level (str)**: "day"/"week"/"month"/"year"/"" (default: all time)\n'
        "- **use_cache (bool)**: True (cached) / False (fresh search, default: True)\n\n"
        "## SEARCH STRATEGY:\n"
        '- Breaking news: time_level="day"\n'
        '- Recent trends: time_level="week"\n'
        '- Monthly/annual data: time_level="month"/"year"\n'
        "- Start with cached search, retry with use_cache=False if results are poor\n"
        '- Cache auto-disabled for "day"/"week" searches\n\n'
        "## YOUR TASK:\n"
        "1. **Research**: Use the tool to gather relevant information\n"
        "2. **Extract & Synthesize**: Identify key facts, connect insights across sources\n"
        "3. **Present**: Organize findings clearly, prioritize relevance, cite sources\n\n"
        "**OUTPUT**: Present the most critical insights first, remove redundant details, and mention conflicting information explicitly.\n\n"
        "**WORKFLOW**: Search → Extract key insights → Present in structured format"
    ),
    name="research_agent",
)
