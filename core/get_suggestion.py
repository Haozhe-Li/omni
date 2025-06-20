from langchain_openai import ChatOpenAI
from core.globalvaris import *

model = ChatOpenAI(model=OPENAI_CHAT_MODEL_FAST).with_structured_output(
    method="json_mode"
)


class SuggestionAgent:
    def __init__(self):
        self.model = model
        self.prompt = """
You are a suggestion agent. Your task is to provide possible follow-up questions based on the provided question.
You MUST use the same language as the question.
You should answer in json mode, follow the schema below:
{{"suggestion": ["suggestion1", "suggestion2", "suggestion3", "suggestion4"]}}
{question}
"""

    def get_suggestion(self, question: str) -> str:
        print(question)
        """Get a suggestion based on the provided question."""
        response = self.model.invoke(self.prompt.format(question=question))
        return response
