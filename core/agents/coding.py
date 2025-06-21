from langgraph.prebuilt import create_react_agent

from langchain_community.tools.riza.command import ExecPython
from core.globalvaris import OPENAI_CHAT_MODEL

model = f"openai:{OPENAI_CHAT_MODEL}"

coding_agent = create_react_agent(
    model=model,
    tools=[ExecPython()],
    prompt=(
        "You are a coding agent. You will assist to write code in python, then run the code and get the output.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with coding tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- Resond your answer in <agent_response> tag\n"
        "- In your response, make sure you include the code you wrote and the output of the code.\n"
        "- Put the code in markdown code block with the language specified as 'python'.\n"
    ),
    name="coding_agent",
)
