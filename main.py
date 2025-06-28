from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.supervisors import supervisor
from core.light_agent import light
from core.utils import pretty_yield_messages, clean_messages, format_tool_messages
import json
from typing import List, Dict

from core.get_suggestion import SuggestionAgent
from core.sources import ss

# allow all cors
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Omni API", description="A REST API for the Omni supervisor system")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class QueryModel(BaseModel):
    messages: List[Dict[str, str]]
    mode: str
    location: str = None  # Optional location field for light agent queries


class SuggestionModel(BaseModel):
    question: str


@app.post("/stream")
async def stream_endpoint(input_query: QueryModel):
    """
    Stream the response from the supervisor system.

    Args:
        input_query: The query data containing the user's messages

    Returns:
        A streaming response with the supervisor's outputs
    """
    # Use messages directly from input_query
    input_data = {"messages": input_query.messages}
    mode = input_query.mode
    location = input_query.location
    print(location)
    if location:
        query_str = input_query.messages[-1]["content"]
        query_str = f"User's current location is {location}. Please use this information for personalization in weather and timing. However, if another specific location is mentioned in the query, use that instead. Below is the query: {query_str}"
        input_data["messages"][-1]["content"] = query_str
    print("Input Data:", input_data)

    activate_agent = light if mode == "light" else supervisor

    async def response_generator():
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
                            # cut summarizing_agent messages
                            message_part = message_part.split(
                                "Name: summarizing_agent"
                            )[-1]
                            message_part = message_part.strip()
                            yield f"data: {json.dumps({'answer': message_part})}\n\n"
                        elif "Name: supervisor" in message_part:
                            message_part = message_part.split("Name: supervisor")[-1]
                            message_part = message_part.strip()
                            yield f"data: {json.dumps({'answer': message_part})}\n\n"
                        elif "Name: light_agent" in message_part:
                            # cut light_agent messages
                            message_part = message_part.split("Name: light_agent")[-1]
                            message_part = message_part.strip()
                            yield f"data: {json.dumps({'answer': message_part})}\n\n"
                        else:
                            message_part = format_tool_messages(message_part)
                            yield f"data: {json.dumps({'tool': message_part})}\n\n"

            # Send a completion message
            sourses = ss.get_sources()
            if sourses:
                yield f"data: {json.dumps({'sources': sourses})}\n\n"
                ss.clear_sources()
            yield f"data: {json.dumps({'content': '[DONE]'})}\n\n"
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            print(error_message)
            yield f"data: {json.dumps({'answer': error_message})}\n\n"

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
    """
    Get a suggestion from the SuggestionAgent.

    Args:
        input_query: The query data containing the user's messages

    Returns:
        A JSON response with the suggestion
    """
    input_data = suggestion_model.question
    suggestion_agent = SuggestionAgent()
    if not input_data or input_data.strip() == "":
        suggestion = suggestion_agent.get_welcome_suggestion()
    else:
        suggestion = suggestion_agent.get_suggestion(question=input_data)
    print("Suggestion:", suggestion)
    return suggestion


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}
