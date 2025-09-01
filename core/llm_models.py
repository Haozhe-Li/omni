import os

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

OPENAI_CHAT_MODEL = "openai/gpt-oss-120b"
OPENAI_CHAT_MODEL_FAST = "openai/gpt-oss-20b"
OPENAI_CHAT_MODEL_ULTRA_FAST = "gpt-4.1-nano"
OPENAI_REASONING_MODEL = "leave_blank"

GROQ_CHAT_MODEL_ULTRA_FAST = "llama-3.1-8b-instant"
GROQ_CHAT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_CHAT_MODEL_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_REASONING_MODEL = "qwen/qwen3-32b"


class LLMModels:
    """
    A class to encapsulate the various LLM models used in the application.
    """

    def __init__(self):
        self.supervisor_model = init_chat_model(f"groq:{OPENAI_CHAT_MODEL}")
        self.research_model = f"openai:{OPENAI_CHAT_MODEL_ULTRA_FAST}"
        self.math_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
        self.web_page_model = f"openai:{OPENAI_CHAT_MODEL_ULTRA_FAST}"
        self.planning_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
        self.timing_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
        self.coding_model = f"openai:{OPENAI_CHAT_MODEL_ULTRA_FAST}"
        self.suggestion_model = ChatGroq(
            model=GROQ_CHAT_MODEL_ULTRA_FAST
        ).with_structured_output(method="json_mode")
        self.summarizing_model = f"groq:{GROQ_REASONING_MODEL}"
        self.weather_model = f"openai:{OPENAI_CHAT_MODEL_ULTRA_FAST}"
        self.light_agent_model = f"groq:{OPENAI_CHAT_MODEL_FAST}"


default_llm_models = LLMModels()
