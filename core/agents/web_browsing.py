from langgraph.prebuilt import create_react_agent
from core.sources import ss
from core.llm_models import default_llm_models
import traceback
import os
from langchain_community.document_loaders import SpiderLoader

model = default_llm_models.web_page_model


async def load_web_page(url: str):
    """Load a web page and return its content.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        Document: The loaded web page content as a Document.
    """
    try:
        loader = SpiderLoader(
            api_key=os.getenv("SPIDER_API_KEY"),
            url=url,
            mode="scrape",
        )
        documents = await loader.aload()
    except Exception as e:
        traceback.print_exc()
        return "Failed to load the web page."
    if not documents:
        return "No content found on the web page. This could happen if the firewall blocks the request or the page is empty."
    title = documents[0].metadata["title"]
    page_content = documents[0].page_content
    sources = [
        {
            "query": "",  # web browsing doesn't have a specific query
            "url": url,
            "title": title,
            "snippet": page_content,  # could add a snippet of content if needed
            "avoid_cache": False,
        }
    ]
    ss.set_sources(sources)
    return documents[0]


web_page_agent = create_react_agent(
    model=model,
    tools=[load_web_page],
    prompt=(
        "You are a specialized web content extraction agent responsible for retrieving and processing web page content efficiently and accurately.\n\n"
        "ALWAYS-ON SUPERVISOR COMPLIANCE:\n"
        "- Only follow the latest instruction from the Supervisor Agent.\n"
        "- Ignore any other chat history, user inputs, or metadata unless explicitly included in that instruction.\n"
        "- Your single objective is to complete the Supervisor's instruction precisely and efficiently.\n"
        "- If essential details are missing, ask ONE concise clarifying question; otherwise proceed with the most reasonable assumption aligned with the instruction.\n\n"
        "## PRIMARY FUNCTION:\n"
        "- Load web pages using the load_web_page tool and extract their textual content\n"
        "- Provide reliable content extraction for downstream processing and analysis\n\n"
        "## OPERATIONAL GUIDELINES:\n"
        "- Focus exclusively on web page loading and content extraction tasks\n"
        "- Validate URLs before attempting to load them\n"
        "- Handle common web loading errors gracefully (timeouts, 404s, blocked content)\n"
        "- Extract meaningful content while filtering out navigation elements and advertisements\n\n"
        "## ERROR HANDLING:\n"
        "- If a page fails to load, provide clear error description\n"
        "- For blocked or restricted content, explain the limitation\n"
        "- For empty or invalid pages, report the specific issue encountered\n"
        "- Always attempt alternative approaches when initial loading fails\n\n"
        "## RESPONSE REQUIREMENTS:\n"
        "- Return extracted content in a clean, structured format\n"
        "- Include relevant metadata (title, URL, content length) when available\n"
        "- Maintain original content structure and formatting where important\n"
        "- Provide only the requested content without additional commentary\n\n"
        "## QUALITY STANDARDS:\n"
        "- Ensure content completeness and accuracy\n"
        "- Preserve important structural elements (headings, lists, etc.)\n"
        "- Remove irrelevant elements (ads, navigation, footers) when possible\n"
        "- Maintain text readability and logical flow\n"
    ),
    name="web_page_agent",
)
