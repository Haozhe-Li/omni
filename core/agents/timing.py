import datetime


def get_current_time(timezone: int) -> str:
    """
    Get the current time in the specified timezone.

    Args:
        timezone (int): The timezone to get the current time for. e.g. 8 for UTC+8, -5 for UTC-5.

    Returns:
        str: The current time in the specified timezone.
    """
    try:
        # Convert the timezone string to a datetime object
        tz = datetime.timezone(datetime.timedelta(hours=timezone))
        # Get the current time in that timezone
        current_time = datetime.datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "Invalid timezone format. Please use an integer offset from UTC."


from langgraph.prebuilt import create_react_agent
from core.globalvaris import GROQ_CHAT_MODEL_FAST

model = f"groq:{GROQ_CHAT_MODEL_FAST}"

timing_agent = create_react_agent(
    model=model,
    tools=[get_current_time],
    prompt=(
        "You are a timing agent. You can calculate current time globally. \n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with time calculating.\n"
        "- When calculating time, make sure you use the correct timezone format.\n"
        "- If you were not given a timezone or a place, you should return you need a timezone or a place to calculate the time.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- Resond your answer in <agent_response> tag\n"
    ),
    name="timing_agent",
)
