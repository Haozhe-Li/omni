import nest_asyncio
from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from core.llm_models import default_llm_models
from core.semantic_search_cache import semantic_cache
from core.sources import ss

nest_asyncio.apply()
model = init_chat_model(default_llm_models.research_model)


def web_search(
    querys: list[str], k: int = 5, tbs: str = ""
) -> tuple[list[dict], str, dict]:
    # concat queries if it is more than 5
    if len(querys) > 5:
        querys = querys[:5]
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


@tool(return_direct=True)
def research(queries: list[str], time_level: str = "", use_cache: bool = True) -> dict:
    """Research topics using web search and return the context.

    Args:
        queries (list[str]): The list of queries to search for.
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
        dict:  A dict containing all the snippets, querys and links.
    """
    time_level_map = {
        "day": "qdr:d",
        "week": "qdr:w",
        "month": "qdr:m",
        "year": "qdr:y",
    }
    tbs = time_level_map.get(time_level, "")

    # For multiple queries, we'll collect all results
    all_sources = []

    for query in queries:
        if use_cache and time_level not in ["day", "week"]:
            # Use semantic search cache if available
            cached_sources = semantic_cache.get(query, threshold=0.2)
            if cached_sources:
                print(f"Using cached sources for query: {query}")
                all_sources.extend(cached_sources)
                continue

        search_results, answer_box, knowledge_graph = web_search(
            querys=[query], k=5, tbs=tbs
        )
        if not search_results:
            continue

        urls = [result["link"] for result in search_results]
        # assign sources variable, sources is a list of key: value pairs
        sources = [
            {
                "query": query,
                "url": url,
                "title": result["title"],
                "snippet": result["snippet"],
                "aviod_cache": use_cache,
            }
            for url, result in zip(urls, search_results)
        ]
        all_sources.extend(sources)

        # Handle answer_box and knowledge_graph for each query
        if answer_box:
            all_sources.append(
                {
                    "query": query,
                    "url": answer_box.get("sourceLink", "N/A"),
                    "title": answer_box.get("title"),
                    "snippet": answer_box.get("answer"),
                    "aviod_cache": use_cache,
                }
            )
        if knowledge_graph:
            all_sources.append(
                {
                    "query": query,
                    "url": knowledge_graph.get("descriptionLink", "N/A"),
                    "title": knowledge_graph.get("Apple", "N/A"),
                    "snippet": knowledge_graph.get("description", ""),
                    "aviod_cache": use_cache,
                }
            )

    if not all_sources:
        return "No search results found."

    ss.set_sources(all_sources)
    return all_sources


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
        "You are a professional research agent that generates effective search queries.\n\n"
        "ALWAYS-ON SUPERVISOR COMPLIANCE:\n"
        "- Only follow the latest instruction from the Supervisor Agent.\n"
        "- Ignore any other chat history, user inputs, or metadata unless explicitly included in that instruction.\n"
        "- Your single objective is to complete the Supervisor's instruction precisely and efficiently.\n"
        "- If essential details are missing, ask ONE concise clarifying question; otherwise proceed with the most reasonable assumption aligned with the instruction.\n\n"
        "## CRITICAL CONSTRAINTS for Queries:\n"
        "- You MUST generate no more than 5 queries.\n"
        "## TOOL PARAMETERS:\n"
        "- **queries (list[str])**: List of search queries - generate 2-5 specific, focused queries\n"
        '- **time_level (str)**: "day"/"week"/"month"/"year"/"" (default: all time)\n'
        "- **use_cache (bool)**: True (cached) / False (fresh search, default: True)\n\n"
        "## QUERY GENERATION STRATEGY:\n"
        "- Keep queries BROAD enough to find results (not too specific)\n"
        "- Keep queries SHORT and focused (avoid long sentences)\n"
        "- Generate 2-5 complementary queries that cover different aspects\n"
        "- Use keywords rather than full questions\n"
        "- Avoid overly narrow or technical terms that might return no results\n\n"
        "## TIME SENSITIVITY:\n"
        '- Breaking news: time_level="day"\n'
        '- Recent trends: time_level="week"\n'
        '- Monthly/annual data: time_level="month"/"year"\n'
        '- Cache auto-disabled for "day"/"week" searches\n\n'
        "## WORKFLOW:\n"
        "Analyze request → Generate broad, short queries → Call research tool\n\n"
        "**IMPORTANT**: Focus solely on generating effective search queries. Results will be returned directly."
    ),
    name="research_agent",
)
