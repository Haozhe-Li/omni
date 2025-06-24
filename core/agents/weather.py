from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models
from langchain_community.utilities import OpenWeatherMapAPIWrapper

model = default_llm_models.weather_model


def get_current_weather(location: str) -> str:
    """
    Get the current weather for a specified location.

    Args:
        location (str): The location to get the current weather for.

    Returns:
        str: A summary of the current weather conditions.
    """
    weather = OpenWeatherMapAPIWrapper()
    return str(weather.run(location))


weather_agent = create_react_agent(
    model=model,
    tools=[get_current_weather],
    prompt=(
        "You are a weather assistant. Your task is to provide accurate weather information for locations.\n\n"
        "INSTRUCTIONS:\n"
        "- Use the get_current_weather tool to fetch weather data for the requested location\n"
        "- Always use English name for locations.\n"
        "- Locations should be followed by a comma and the country name (e.g., 'New York, USA')\n"
        "- Provide concise summaries of current weather conditions\n"
        "- If the location is unclear, ask for clarification\n"
        "- Present temperature, conditions, humidity, and wind information when available\n"
        "- Respond ONLY with relevant weather information\n"
        "- Respond your answer in <agent_response> tag\n"
    ),
    name="weather_agent",
)
