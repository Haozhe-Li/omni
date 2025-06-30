from langchain.chat_models import init_chat_model
import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

OPENAI_CHAT_MODEL = "gpt-4.1"
OPENAI_CHAT_MODEL_FAST = "gpt-4.1-mini"
OPENAI_CHAT_MODEL_ULTRA_FAST = "gpt-4.1-nano"
OPENAI_REASONING_MODEL = "o"

GROQ_CHAT_MODEL_ULTRA_FAST = "llama-3.1-8b-instant"
GROQ_CHAT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_CHAT_MODEL_FAST = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_REASONING_MODEL = "qwen/qwen3-32b"


class LLMModels:
    def __init__(self):
        preset = os.getenv("LLM_MODEL_PRESET", "groq")
        if preset == "openai":
            print("Using OpenAI models")
            self.supervisor_model = init_chat_model(f"openai:{OPENAI_CHAT_MODEL}")
            self.research_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.math_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.web_page_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.planning_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.timing_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.coding_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.suggestion_model = ChatOpenAI(
                model=OPENAI_CHAT_MODEL_ULTRA_FAST
            ).with_structured_output(method="json_mode")
            self.summarizing_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.weather_model = f"openai:{OPENAI_CHAT_MODEL_ULTRA_FAST}"
            self.light_agent_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
        else:
            print("Using Groq models")
            self.supervisor_model = init_chat_model(f"openai:{OPENAI_CHAT_MODEL}")
            self.research_model = f"openai:{OPENAI_CHAT_MODEL_FAST}"
            self.math_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
            self.web_page_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
            self.planning_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
            self.timing_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
            self.coding_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
            self.suggestion_model = ChatGroq(
                model=GROQ_CHAT_MODEL_ULTRA_FAST
            ).with_structured_output(method="json_mode")
            self.summarizing_model = f"groq:{GROQ_REASONING_MODEL}"
            self.weather_model = f"groq:{GROQ_CHAT_MODEL_FAST}"
            self.light_agent_model = f"groq:{GROQ_REASONING_MODEL}"


default_llm_models = LLMModels()
