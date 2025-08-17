from langchain_community.document_loaders import WebBaseLoader
from langgraph.prebuilt import create_react_agent
from core.sources import ss
from core.llm_models import default_llm_models

model = default_llm_models.web_page_model


def load_web_page(url: str):
    """Load a web page and return its content."""
    loader = WebBaseLoader(url)
    documents = loader.load()
    if not documents:
        return "No content found on the web page."
    title = documents[0].metadata["title"]
    sources = [
        {
            "query": "",  # web browsing doesn't have a specific query
            "url": url,
            "title": title,
            "snippet": "",  # could add a snippet of content if needed
            "from_cache": False,
        }
    ]
    ss.set_sources(sources)
    return documents[0]


web_page_agent = create_react_agent(
    model=model,
    tools=[load_web_page],
    prompt=(
        "You are a web page agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with web page loading tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- You must respond your answer in <agent_response> tag\n"
    ),
    name="web_page_agent",
)
