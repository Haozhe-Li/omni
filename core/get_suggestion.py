from core.llm_models import default_llm_models


class SuggestionAgent:
    """A class to generate suggestions for user queries."""

    def __init__(self):
        self.model = default_llm_models.suggestion_model
        self.welcome_prompt = """
You are an intelligent suggestion agent. Generate 4 engaging and diverse questions that would interest a new user exploring this AI assistant.

CRITICAL REQUIREMENTS:
- Generate exactly 4 suggestions
- Make questions diverse across different categories (news, capabilities, weather, people, technology, etc.)
- Keep questions concise and natural
- Use proper grammar and engaging language
- Return ONLY valid JSON in the exact format specified

JSON Schema:
{{"suggestion": ["question1", "question2", "question3", "question4"]}}

Example Output:
{{"suggestion": ["What's happening in the world today?", "What capabilities do you have?", "What's the weather like in my area?", "Tell me about recent AI developments"]}}
"""
        self.prompt = """
You are an intelligent suggestion agent. Generate 4 relevant follow-up questions based on the user's original question.

CRITICAL REQUIREMENTS:
- MUST use the exact same language as the original question
- Generate exactly 4 suggestions
- Make suggestions naturally build upon or relate to the original question
- Provide diverse angles: deeper details, related topics, comparisons, next steps
- Keep questions concise and conversational
- Return ONLY valid JSON in the exact format specified

JSON Schema:
{{"suggestion": ["question1", "question2", "question3", "question4"]}}

Example:
Original Question: "How's the weather in New York?"
Output:
{{"suggestion": ["Will it rain in New York today?", "What's the temperature forecast for this week?", "How does New York weather compare to Los Angeles?", "What should I wear for today's weather?"]}}

Original Question: {question}
Output:"""

    async def get_suggestion(self, question: str) -> str:
        """Get a suggestion based on the provided question.

        Args:
            question (str): The user's question.

        Returns:
            str: A JSON string containing the suggested follow-up questions.
        """
        return await self.model.ainvoke(self.prompt.format(question=question))

    async def get_welcome_suggestion(self) -> str:
        """Get a welcome suggestion.

        Returns:
            str: A JSON string containing the welcome suggestion.
        """
        return await self.model.ainvoke(self.welcome_prompt)


suggestion_agent = SuggestionAgent()
