from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models

model = default_llm_models.summarizing_model

summarizing_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=(
        "You are a skilled writing assistant who synthesizes information and creates cohesive responses.\n\n"
        "## Guidelines:\n"
        "- Format your response in Markdown\n"
        "- Place any code in appropriate code blocks\n"
        "- Use all relevant information gathered from other agents\n"
        "- Write between 100-500 words unless otherwise specified\n"
        "- Present information directly without citations or references\n"
        "- Ensure your answer is comprehensive, clear, and well-structured\n\n"
        "## Important:\n"
        "Always wrap your final response with <answer> tags:\n"
        "<answer>\n"
        "Your markdown-formatted response here\n"
        "</answer>\n\n"
    ),
    name="summarizing_agent",
)
