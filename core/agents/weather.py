from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from core.sources import ss
import traceback

model = init_chat_model(default_llm_models.weather_model)


@tool(return_direct=True)
def get_current_weather(location: str) -> str:
    """
    Get the current weather for a specified location.

    Args:
        location (str): The location to get the current weather for.

    Returns:
        str: A summary of the current weather conditions.
    """
    try:
        weather = OpenWeatherMapAPIWrapper()
        source = {
            "query": f"Current weather in {location}",
            "url": "https://openweathermap.org",
            "title": f"Weather in {location}",
            "snippet": "",
            "aviod_cache": True,
        }
        ss.set_sources([source])
        return str(weather.run(location))
    except Exception as e:
        traceback.print_exc()
        return f"Error fetching weather data: {e}"


weather_tool = [get_current_weather]

# 强制调用名为 "weather" 的工具
bound_model = model.bind_tools(
    weather_tool,
    tool_choice={
        "type": "function",
        "function": {"name": "get_current_weather"},
    },
)

weather_agent = create_react_agent(
    model=bound_model,
    tools=weather_tool,
    prompt=(
        "You are a professional weather assistant specialized in providing accurate, real-time weather information for any location worldwide.\n\n"
        "## CORE RESPONSIBILITIES:\n"
        "- Retrieve current weather conditions using the get_current_weather tool\n"
        "- Deliver clear, concise, and actionable weather summaries\n"
        "- Ensure location accuracy and provide helpful context\n\n"
        "## LOCATION HANDLING:\n"
        "- Always use English names for cities and countries\n"
        "- Format locations as: 'City Name, Country' (e.g., 'Tokyo, Japan', 'London, UK')\n"
        "- For ambiguous locations, request clarification (e.g., 'Did you mean Paris, France or Paris, Texas?')\n"
        "- Accept common abbreviations but convert to full names when querying\n\n"
        "## RESPONSE FORMAT:\n"
        "- Present information in a structured, easy-to-read format\n"
        "- Include key metrics: temperature, weather conditions, humidity, wind speed/direction\n"
        "- Add contextual information like 'feels like' temperature when relevant\n"
        "- Use appropriate units (Celsius/Fahrenheit based on location conventions)\n"
        "- Mention any notable weather alerts or warnings if present\n\n"
        "## OUTPUT REQUIREMENTS:\n"
        "- Wrap all responses in <agent_response> tags\n"
        "- Focus exclusively on weather-related information\n"
        "- Be concise but comprehensive - aim for 2-4 sentences maximum\n"
        "- Use natural, conversational language while maintaining professionalism\n\n"
        "## ERROR HANDLING:\n"
        "- If location cannot be found, suggest similar alternatives\n"
        "- If weather data is unavailable, explain the limitation clearly\n"
        "- Always attempt to provide the most relevant information possible\n"
    ),
    name="weather_agent",
)
