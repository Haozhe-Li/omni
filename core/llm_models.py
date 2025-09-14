import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

GPT_OSS_120B = "groq:openai/gpt-oss-120b"
GPT_OSS_20B = "groq:openai/gpt-oss-20b"
GPT_4_1_NANO = "openai:gpt-4.1-nano"
GPT_4_1 = "openai:gpt-4.1"
GPT_5_1_NANO = "openai:gpt-5-nano-2025-08-07"

LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"
LLAMA_4_MAVERICK = "groq:meta-llama/llama-4-maverick-17b-128e-instruct"
LLAMA_4_SCOUT = "groq:meta-llama/llama-4-scout-17b-16e-instruct"
QWEN_32B = "groq:qwen/qwen3-32b"


class LLMModels:
    """
    A class to encapsulate the various LLM models used in the application.
    """

    def __init__(self):
        self.supervisor_model = GPT_OSS_20B
        self.research_model = GPT_4_1_NANO
        self.web_page_model = GPT_4_1_NANO
        self.coding_model = GPT_OSS_120B
        self.suggestion_model = ChatGroq(
            model=LLAMA_3_1_8B_INSTANT,
            api_key=os.getenv("GROQ_API_KEY"),
        ).with_structured_output(method="json_mode")
        self.summarizing_model = QWEN_32B
        self.weather_model = GPT_4_1_NANO
        self.light_agent_model = init_chat_model(GPT_OSS_20B, reasoning_effort="low")


default_llm_models = LLMModels()
