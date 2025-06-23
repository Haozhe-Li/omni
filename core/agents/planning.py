from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models

model = default_llm_models.planning_model

planning_agent = create_react_agent(
    model=model,
    tools=[],
    prompt=(
        "You are a planning agent. Your task is to think about the question and plan what steps to take next.\n\n"
        "For example, if the question is 'How's the weather?', you might plan to:\n"
        "First search the current weather online.\n"
        "Then summarize the weather conditions.\n"
        "Finally, respond to the supervisor with the weather summary.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with planning tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- Planning ONLY, DO NOT answer any questions directly.\n"
        "- Respond your answer in <agent_response> tag\n"
    ),
    name="planning_agent",
)
