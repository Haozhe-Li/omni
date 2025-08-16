from langchain_community.utilities import GoogleSerperAPIWrapper
import nest_asyncio
from core.llm_models import default_llm_models
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from functools import lru_cache

model = init_chat_model(default_llm_models.research_model)

from langchain_community.document_loaders import WebBaseLoader
from core.sources import ss

nest_asyncio.apply()


def load_web_page(urls: list[str]) -> str:
    """Load a web page and return its content.

    Args:
        urls (list[str]): A list of URLs to load.

    Returns:
        str: The content of the web page.
    """
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents


@lru_cache(maxsize=128)
def web_search(
    querys: list[str], k: int = 3, tbs: str = ""
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


@tool(return_direct=True)
def research(query: str, time_level: str = "") -> str:
    """Research a topic using web search and return the context.

    Args:
        query (str): The query to search for.
        time_level (str): The time level for the search. For most of the time you would not use this parameter.
                          Available options are "day", "week", "month", "year".
                          Not passing this parameter will use the default which search results for anytime.
                          You will only use this when the search is time sensitive enough.
                          Defaults to "" means anytime.

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
    search_results, answer_box, knowledge_graph = web_search(
        querys=[query], k=5, tbs=tbs
    )
    if not search_results:
        return "No search results found."
    urls = [result["link"] for result in search_results]
    # assign sources variable, sources is a list of key: value pairs
    sources = [
        {"url": url, "title": result["title"], "snippet": result["snippet"]}
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
                "snippet": "",
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
                "snippet": "",
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
        "You are a professional research agent specialized in conducting comprehensive web searches and information gathering.\n\n"
        "## CORE RESPONSIBILITIES:\n"
        "- Conduct thorough web searches using the `research` tool\n"
        "- Gather, synthesize, and present accurate, up-to-date information\n"
        "- Provide comprehensive coverage of research topics\n"
        "- Ensure information quality and relevance\n\n"
        "## SEARCH STRATEGY:\n"
        "1. **Progressive Refinement**: Start with broad queries, then narrow down to specific aspects\n"
        "2. **Multiple Perspectives**: Search from different angles to get comprehensive coverage\n"
        "3. **Current Information**: Use time_level parameter when searching for recent developments\n"
        "4. **Quality Sources**: Focus on authoritative and credible information sources\n\n"
        "## QUERY EXAMPLES:\n"
        "- For 'AI in healthcare': \n"
        "  → 'artificial intelligence healthcare applications'\n"
        "  → 'AI medical diagnosis current trends'\n"
        "  → 'machine learning clinical decision support systems'\n"
        "- For 'renewable energy': \n"
        "  → 'renewable energy sources 2025'\n"
        "  → 'solar wind energy efficiency comparison'\n"
        "  → 'renewable energy policy global initiatives'\n\n"
        "## SEARCH GUIDELINES:\n"
        "- Use clear, specific keywords that capture the essence of the topic\n"
        "- Include relevant synonyms and alternative terms\n"
        "- Consider temporal aspects (recent developments, historical context)\n"
        "- Search for both general overviews and specific implementations\n"
        "- Look for statistical data, case studies, and expert opinions\n\n"
        "## TIME-SENSITIVE SEARCHES:\n"
        "Use the time_level parameter for:\n"
        "- Breaking news or current events (day/week)\n"
        "- Recent technological developments (month)\n"
        "- Annual reports or yearly trends (year)\n\n"
        "## STRICT LIMITATIONS:\n"
        "- ONLY perform research-related tasks\n"
        "- DO NOT engage in conversations outside of research scope\n"
        "- DO NOT provide personal opinions or speculation\n"
        "- Present ONLY factual information from search results\n\n"
        "## OUTPUT FORMAT:\n"
        "- Respond directly with research findings\n"
        "- Include relevant context and key insights\n"
        "- Maintain objectivity and accuracy\n"
        "- NO additional commentary or personal remarks"
    ),
    name="research_agent",
)
