from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.supervisors import supervisor
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

    async def response_generator():
        try:

            # Stream responses from supervisor
            for chunk in supervisor.stream(input_data):
                # Use pretty_yield_messages to format output
                for message_part in pretty_yield_messages(chunk, last_message=True):
                    # Format each message part as a server-sent event
                    message_part = clean_messages(message_part)
                    if message_part:
                        print(message_part)
                        if "<answer>" in message_part:
                            # cut only between <answer> and </answer>
                            start = message_part.find("<answer>") + len("<answer>")
                            end = message_part.find("</answer>")
                            message_part = message_part[start:end].strip()
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
            yield f"data: {json.dumps({'error': error_message})}\n\n"

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
