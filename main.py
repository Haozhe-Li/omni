import json
from dotenv import load_dotenv

load_dotenv()
from typing import List, Dict
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.supervisors import supervisor
from core.light_agent import light
from core.utils import pretty_yield_messages, clean_messages, format_tool_messages
from core.get_suggestion import SuggestionAgent
from core.sources import ss
from core.semantic_search_cache import semantic_cache
import traceback

load_dotenv()
app = FastAPI(title="Omni API", description="A REST API for the Omni supervisor system")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class QueryModel(BaseModel):
    """Model representing a user query.

    Args:
        BaseModel (pydantic.BaseModel): The base model class from Pydantic.
    """

    messages: List[Dict[str, str]]
    mode: str
    location: str = None  # Optional location field for light agent queries
    preferredLanguage: str = None  # Optional preferred language field


class SuggestionModel(BaseModel):
    """Model representing a suggestion.

    Args:
        BaseModel (pydantic.BaseModel): The base model class from Pydantic.
    """

    question: str


@app.post("/stream")
async def stream_endpoint(input_query: QueryModel) -> StreamingResponse:
    """
    Stream responses from the supervisor system based on the user query.

    Args:
        input_query (QueryModel): The query data containing the user's messages, mode, location,
                                  and preferred language.

    Returns:
        StreamingResponse: A streaming response with the supervisor's outputs.
    """
    input_data = {"messages": input_query.messages}
    mode = input_query.mode
    location = input_query.location
    preferred_language = input_query.preferredLanguage
    system_str = ""
    if location:
        system_str += f"""
        ## Location Personalization:
        User's current location is {location}. Please use this information for personalization in weather and timing. 
        However, if another specific location is mentioned in the query, use that instead."""
    if preferred_language:
        system_str += f"""
        ## CRITICAL LANGUAGE REQUIREMENT:
        "**ALWAYS respond in the SAME language as users preference language {preferred_language}** This is absolutely critical.
        The user will only understand your response if it matches their language.
        """
    else:
        system_str += """
        ## CRITICAL LANGUAGE REQUIREMENT:\n"
        "**ALWAYS respond in the SAME language as the user's input.** This is absolutely critical:\n"
        "- If user writes in English → respond in English\n"
        "- If user writes in Chinese → respond in Chinese\n"
        "- If user writes in any other language → respond in that language\n"
        "The user will only understand your response if it matches their language.\n\n"""
    if system_str:
        input_data["messages"].insert(0, {"role": "system", "content": system_str})

    print("Input Data:", input_data)

    activate_agent = light if mode == "light" else supervisor

    async def response_generator():
        """Generate a streaming response from the supervisor system.

        Yields:
            str: A server-sent event containing the supervisor's output.
        """
        try:
            # Stream responses from supervisor
            for chunk in activate_agent.stream(input_data):
                # Use pretty_yield_messages to format output
                for message_part in pretty_yield_messages(chunk, last_message=True):
                    # Format each message part as a server-sent event
                    message_part = clean_messages(message_part)
                    if message_part:
                        print(message_part)
                        if "Name: summarizing_agent" in message_part:
                            message_part = message_part.split(
                                "Name: summarizing_agent"
                            )[-1]
                            # extract part in <think></think>
                            thinking_part = message_part.split("<think>")[-1]
                            thinking_part = thinking_part.split("</think>")[0]
                            yield f"data: {json.dumps({'tool': thinking_part})}\n\n"
                            message_part = message_part.split("</think>")[-1].strip()
                            yield f"data: {json.dumps({'answer': message_part})}\n\n"
                        elif "Name: supervisor" in message_part:
                            message_part = message_part.split("Name: supervisor")[-1]
                            thinking_part = message_part.split("<think>")[-1]
                            thinking_part = thinking_part.split("</think>")[0]
                            yield f"data: {json.dumps({'tool': thinking_part})}\n\n"
                            message_part = message_part.split("</think>")[-1].strip()
                            yield f"data: {json.dumps({'answer': message_part})}\n\n"
                        elif "Name: light_agent" in message_part:
                            # cut light_agent messages
                            message_part = message_part.split("Name: light_agent")[-1]
                            thinking_part = message_part.split("<think>")[-1]
                            thinking_part = thinking_part.split("</think>")[0]
                            # yield f"data: {json.dumps({'tool': thinking_part})}\n\n"
                            message_part = message_part.split("</think>")[-1].strip()
                            yield f"data: {json.dumps({'answer': message_part})}\n\n"
                        else:
                            message_part = format_tool_messages(message_part)
                            yield f"data: {json.dumps({'tool': message_part})}\n\n"

            # Send a completion message
            sourses = ss.get_sources()
            if sourses:
                yield f"data: {json.dumps({'sources': sourses})}\n\n"
                semantic_cache.add(sources=sourses)
                ss.clear_sources()
            yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
        except Exception as e:
            traceback.print_exc()
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            res_error = """Sorry, something went wrong while processing your request. Please try again later."""
            yield f"data: {json.dumps({'answer': res_error})}\n\n"

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@app.post("/suggestion")
async def suggest_endpoint(suggestion_model: SuggestionModel):
    input_data = suggestion_model.question
    suggestion_agent = SuggestionAgent()
    if not input_data or input_data.strip() == "":
        suggestion = suggestion_agent.get_welcome_suggestion()
    else:
        suggestion = suggestion_agent.get_suggestion(question=input_data)
    print("Suggestion:", suggestion)
    return suggestion


@app.get("/health")
async def health_check() -> dict:
    """Check the health status of the service.

    Returns:
        dict: A dictionary containing the health status.
    """
    return {"status": "healthy"}
