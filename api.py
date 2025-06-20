from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.supervisors import supervisor
from core.utils import pretty_yield_messages, clean_messages
import json

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
    query: str


@app.post("/stream")
async def stream_endpoint(query_data: QueryModel):
    """
    Stream the response from the supervisor system.

    Args:
        query_data: The query data containing the user's message

    Returns:
        A streaming response with the supervisor's outputs
    """

    async def response_generator():
        try:
            # Create messages structure expected by supervisor
            input_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": query_data.query,
                    }
                ]
            }

            # Stream responses from supervisor
            for chunk in supervisor.stream(input_data):
                # Use pretty_yield_messages to format output
                for message_part in pretty_yield_messages(chunk, last_message=True):
                    # Format each message part as a server-sent event
                    message_part = clean_messages(message_part)
                    if message_part:
                        if "<answer>" in message_part:
                            # cut only between <answer> and </answer>
                            start = message_part.find("<answer>") + len("<answer>")
                            end = message_part.find("</answer>")
                            message_part = message_part[start:end].strip()
                            yield f"data: {json.dumps({'answer': message_part})}\n\n"
                        else:
                            yield f"data: {json.dumps({'tool': message_part})}\n\n"

            # Send a completion message
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


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
