from langchain_community.document_loaders import WebBaseLoader
from langgraph.prebuilt import create_react_agent


def load_web_page(url: str):
    """Load a web page and return its content."""
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents[0]


from core.globalvaris import OPENAI_CHAT_MODEL_FAST

model = f"openai:{OPENAI_CHAT_MODEL_FAST}"

web_page_agent = create_react_agent(
    model=model,
    tools=[load_web_page],
    prompt=(
        "You are a web page agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with web page loading tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- Resond your answer in <agent_response> tag\n"
    ),
    name="web_page_agent",
)
