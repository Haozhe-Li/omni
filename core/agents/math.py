from langgraph.prebuilt import create_react_agent
from core.globalvaris import GROQ_CHAT_MODEL_FAST

model = f"groq:{GROQ_CHAT_MODEL_FAST}"

from langchain_community.tools.riza.command import ExecPython

math_agent = create_react_agent(
    model=model,
    tools=[ExecPython()],
    prompt=(
        "You are a math agent. You can write python equation in python, then run the equation to get result.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with coding tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- You must respond your answer in <agent_response> tag\n"
    ),
    name="math_agent",
)
