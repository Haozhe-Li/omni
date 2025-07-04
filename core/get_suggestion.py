from core.llm_models import default_llm_models


class SuggestionAgent:
    """A class to generate suggestions for user queries."""

    def __init__(self):
        self.model = default_llm_models.suggestion_model
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
        """Get a suggestion based on the provided question.

        Args:
            question (str): The user's question.

        Returns:
            str: A JSON string containing the suggested follow-up questions.
        """
        response = self.model.invoke(self.prompt.format(question=question))
        return response

    def get_welcome_suggestion(self) -> str:
        """Get a welcome suggestion.

        Returns:
            str: A JSON string containing the welcome suggestion.
        """
        response = self.model.invoke(self.welcome_prompt)
        return response
