from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from core.globalvaris import *

model = ChatOpenAI(model=OPENAI_CHAT_MODEL_FAST).with_structured_output(
    method="json_mode"
)

groq_model = ChatGroq(model=GROQ_CHAT_MODEL_FAST).with_structured_output(
    method="json_mode"
)


class SuggestionAgent:
    def __init__(self):
        self.model = groq_model
        self.welcome_prompt = """
You are a suggestion agent. Your task is to generate 4 questions that user might wanna be interested in.
You MUST use the same language as the question.
You should answer in json mode, follow the schema below:
{{"suggestion": ["suggestion1", "suggestion2", "suggestion3", "suggestion4"]}}

Example:
suggestion: 
{{"suggestion": ["Any news today?", "What can you do?", "Will it be sunny today?", "Who is Elon Musk?"]}}
"""
        self.prompt = """
You are a suggestion agent. Your task is to provide possible follow-up questions based on the provided question.
You MUST use the same language as the question.
You should answer in json mode, follow the schema below:
{{"suggestion": ["suggestion1", "suggestion2", "suggestion3", "suggestion4"]}}

Example:
question: How's the weather in New York?
suggestion: 
{{"suggestion": ["Is it going to rain in New York today?", "What are the weather conditions in New York right now?", "How does the weather in New York compare to other cities?"]}}

Here's the question:
{question}
"""

    def get_suggestion(self, question: str) -> str:
        print(question)
        """Get a suggestion based on the provided question."""
        response = self.model.invoke(self.prompt.format(question=question))
        return response

    def get_welcome_suggestion(self) -> str:
        """Get a welcome suggestion."""
        response = self.model.invoke(self.welcome_prompt)
        return response
