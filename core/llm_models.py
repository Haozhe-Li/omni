import os

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

GPT_OSS_120B = "openai/gpt-oss-120b"
GPT_OSS_20B = "openai/gpt-oss-20b"
GPT_4_1_NANO = "gpt-4.1-nano"

LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"
LLAMA_4_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
QWEN_32B = "qwen/qwen3-32b"


class LLMModels:
    """
    A class to encapsulate the various LLM models used in the application.
    """

    def __init__(self):
        self.supervisor_model = init_chat_model(
            f"groq:{GPT_OSS_120B}", reasoning_effort="high"
        )
        self.research_model = f"groq:{GPT_OSS_20B}"
        self.math_model = f"groq:{LLAMA_4_SCOUT}"
        self.web_page_model = f"openai:{GPT_4_1_NANO}"
        self.coding_model = f"openai:{GPT_4_1_NANO}"
        self.suggestion_model = ChatGroq(
            model=LLAMA_3_1_8B_INSTANT
        ).with_structured_output(method="json_mode")
        self.summarizing_model = f"groq:{QWEN_32B}"
        self.weather_model = f"openai:{GPT_4_1_NANO}"
        self.light_agent_model = init_chat_model(
            f"groq:{GPT_OSS_20B}", reasoning_effort="low"
        )


default_llm_models = LLMModels()
