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


def web_search(querys: list[str]) -> str:
    """Perform a web search using Google Serper API."""
    print(f"Performing web search for queries: {querys}")
    search = GoogleSerperAPIWrapper(k=3)
    results = []
    for query in querys:
        result = search.results(query)
        results.extend(result.get("organic", []))
    answer_box = result.get("answerBox", "")
    knowledge_graph = result.get("knowledgeGraph", {})
    return results, answer_box, knowledge_graph


@tool(return_direct=True)
def research(querys: list[str]) -> str:
    """Research a topic using web search and return the context.

    Args:
        querys (list[str]): A list of search queries.

    Returns:
        str:  A summary of the search results.
    """
    search_results, answer_box, knowledge_graph = web_search(
        querys[:5]
    )  # Limit to the first 5 queries
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
        "You are a research agent. You could call `research` to perform a web search.\n\n"
        "INSTRUCTIONS:\n"
        "- When calling the `research` tool, provide a list of search queries. Queries should no less than 3 and no more than 5\n"
        "- The queries should be from broad to specific.\n"
        "- For example, if the user asks about 'climate change', you might start with 'climate change', then 'effects of climate change', and finally 'climate change impact on polar bears'. and so on\n"
        "- Assist ONLY with research-related tasks, DO NOT do anything else.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)
