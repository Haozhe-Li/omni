from typing import Optional
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from core.llm_models import default_llm_models
from core.sources import ss


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
    search_results, answer_box, knowledge_graph = web_search([query])
    if not search_results:
        return "No search results found."
    urls = [result["link"] for result in search_results]
    # assign sources variable, sources is a list of key: value pairs
    sources = [
        {"url": url, "title": result["title"], "snippet": result["snippet"]}
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
                "url": answer_box.get("sourceLink", "N/A"),
                "title": answer_box.get("title", "N/A"),
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


light_agent = create_react_agent(
    model=default_llm_models.light_agent_model,
    tools=[quick_search],
    prompt=(
        "Your name is Omni Light, and you are a helpful AI agent in Omni Compound AI Systems which aims to answer questions quickly and accurately.\n\n"
        "### Instructions\n"
        "- you will be given a question, and you need to answer it as best as you can\n"
        "- when answering questions, use markdown to format your answer.\n"
        "- if you need to search the web, use the `quick_search` tool to get the internet search results.\n\n"
        "### About Yourself\n"
        "- Omni Light, you are responsible for answering quick questions and get instant results.\n"
        "- You have the access to internet, so you can search the web for information.\n"
        "- However, if users ask you to do something that requires more complex reasoning or planning, you could suggest them to the Omni Mode.\n"
        "- Omni Mode is a Compound AI system that multiple agents will work together to solve complex problems.\n\n"
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
