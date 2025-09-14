import json
import os
import traceback
import re
from typing import List, Dict, Iterable, Tuple
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.supervisors import supervisor
from core.light_agent import light
from core.utils import pretty_yield_messages
from core.get_suggestion import suggestion_agent
from core.sources import ss
from core.semantic_search_cache import semantic_cache
from core.trie import autocomplete_trie
from core.agents.summarizing import (
    question_answering_agent,
    QUESTION_ANSWERING_SYS_PROMPT,
)

app = FastAPI(title="Omni API", description="A REST API for the Omni supervisor system")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

is_ingest_cache = os.getenv("INGEST_CACHE", "true").lower() == "true"


class QueryModel(BaseModel):
    """Model representing a user query.

    Args:
        BaseModel (pydantic.BaseModel): The base model class from Pydantic.
    """

    messages: List[Dict[str, str]]
    mode: str
    location: str = None  # Optional location field for light agent queries
    preferredLanguage: str = None  # Optional preferred language field
    useCache: bool = True  # Optional field to use cache, default is True
    collectDataToCache: bool = (
        True  # Optional field to collect data to cache, default is True
    )
    dateTime: str = None  # Optional field for date and time, default is None


class SuggestionModel(BaseModel):
    """Model representing a suggestion.

    Args:
        BaseModel (pydantic.BaseModel): The base model class from Pydantic.
    """

    question: str


class AutocompleteQueryModel(BaseModel):
    """Model for autocomplete requests.

    Args:
        BaseModel (pydantic.BaseModel): The base model class from Pydantic.
    """

    prefix: str
    max_suggestions: int = 10


class AutocompleteUpdateModel(BaseModel):
    """Model for updating word frequency in trie.

    Args:
        BaseModel (pydantic.BaseModel): The base model class from Pydantic.
    """

    word: str
    increment: int = 1


class AutocompleteLoadModel(BaseModel):
    """Model for loading data from text file.

    Args:
        BaseModel (pydantic.BaseModel): The base model class from Pydantic.
    """

    file_path: str


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
    use_cache = input_query.useCache
    collect_data_to_cache = input_query.collectDataToCache
    system_str = ""
    date_time = input_query.dateTime
    if location:
        system_str += f"""
        ## Location Personalization:
        User's current location is {location}. Please use this information for personalization in weather and timing. 
        However, if another specific location is mentioned in the query, use that instead."""
    if date_time:
        system_str += f"""
        ## Date and Time Personalization:
        User's current date and time, timezone is {date_time}."""
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
    use_cache = (
        True if activate_agent == light else use_cache
    )  # hardcoded light agent always use cache
    semantic_cache.set_cache_settings(
        useCache=use_cache,
        collectDataToCache=(collect_data_to_cache and is_ingest_cache),
    )

    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _strip_ansi(s: str) -> str:
        return ANSI_RE.sub("", s)

    def _clean_header(text: str) -> str:
        """Remove decorative lines, banners (Ai/Human Message), ANSI codes, blank lines."""
        text = _strip_ansi(text)
        cleaned: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if re.match(r"^=+$", line):
                continue
            if re.search(r"Ai Message", line, re.IGNORECASE):
                continue
            if re.search(r"Human Message", line, re.IGNORECASE):
                continue
            if re.search(r"Tool Message", line, re.IGNORECASE):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _extract_agent_name(text: str) -> Tuple[str | None, str]:
        """Return (agent_name, remaining_text_without_name_line).

        Additional rules:
        - delegation_instruction blocks are attributed to supervisor
        - human messages mapped to supervisor
        """
        # delegation instruction -> supervisor
        stripped = text.lstrip()
        if stripped.startswith("<delegation_instruction"):
            return "supervisor_agent", text
        match = re.search(r"Name:\s*([A-Za-z0-9_\-]+)", text)
        if match:
            agent = match.group(1)
            remaining = re.sub(
                r"Name:\s*([A-Za-z0-9_\-]+)\s*", "", text, count=1
            ).strip()
            # Map human -> supervisor for consistency with frontend expectations
            if agent == "human":
                agent = "supervisor_agent"
            # Map research -> research_agent for consistency
            if agent == "research":
                agent = "research_agent"
            return agent, remaining
        if "Human Message" in text:
            return "supervisor_agent", text
        return None, text

    def _split_summarizing(text: str) -> Iterable[Tuple[str, str]]:
        """Special handling for question_answering_agent containing <think> tags.

        Yields tuples of (key, value). First the thinking part with key 'question_answering_agent',
        then the final answer with key 'answer'. If no think tags, fall back to single event.
        """
        if "<think>" in text and "</think>" in text:
            think_part = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            answer_part = text.split("</think>", 1)[1].strip()
            if think_part:
                yield ("question_answering_agent", think_part)
            if answer_part:
                yield ("answer", answer_part)
        else:
            yield ("question_answering_agent", text.strip())

    def _normalize_event(agent: str | None, content: str) -> Iterable[Tuple[str, str]]:
        """Normalize a single raw message part into one or more (key, value) events."""
        # Clean header decorations
        content = _clean_header(content)
        # If agent missing, detect delegation instruction after cleaning
        if (agent is None or agent == "content") and content.startswith(
            "<delegation_instruction"
        ):
            agent = "supervisor_agent"
        if agent == "question_answering_agent":
            yield from _split_summarizing(content)
        else:
            key = agent if agent else "content"
            yield (key, content.strip())

    async def response_generator():
        """Generate standardized streaming response events per new front-end spec."""
        try:
            collected_results = []
            last_light_agent: str | None = None

            async for chunk in activate_agent.astream(input_data):
                for raw in pretty_yield_messages(chunk, last_message=True):
                    agent, remainder = _extract_agent_name(raw)
                    for key, value in _normalize_event(agent, remainder):
                        if not value:
                            continue

                        if value.strip():
                            collected_results.append({"agent": key, "content": value})

                        if key == "light_agent":
                            # Buffer until we know if it's the final one
                            if last_light_agent is not None:
                                # Previous buffered one is not final, emit as normal
                                yield f"data: {json.dumps({'light_agent': last_light_agent}, ensure_ascii=False)}\n\n"
                            last_light_agent = value
                        else:
                            # Flush buffered light_agent if present (since a different agent appeared, it's not final)
                            if last_light_agent is not None:
                                yield f"data: {json.dumps({'light_agent': last_light_agent}, ensure_ascii=False)}\n\n"
                                last_light_agent = None
                            # Handle supervisor key as intermediate info
                            if key == "supervisor":
                                yield f"data: {json.dumps({'supervisor_agent': value}, ensure_ascii=False)}\n\n"
                            else:
                                yield f"data: {json.dumps({key: value}, ensure_ascii=False)}\n\n"

            # After streaming all chunks, if there is a remaining light_agent message, treat it as final answer
            if last_light_agent is not None:
                collected_results.append(
                    {"agent": "light_agent", "content": last_light_agent}
                )
                yield f"data: {json.dumps({'answer': last_light_agent}, ensure_ascii=False)}\n\n"

            # 如果有收集到的结果，且不是 light_agent 模式，使用 question_answering_agent 进行总结
            if collected_results and activate_agent != light:
                # 构建总结的输入
                summary_content = "Summary Context:\n\n"
                # add user's original question into summary_content, only the last query but not not whole history from input_data
                user_questions = [
                    msg["content"]
                    for msg in input_data["messages"]
                    if msg["role"] == "user"
                ]
                if user_questions:
                    summary_content += f"User's question: {user_questions[-1]}\n\n"
                summary_content += "Here are the findings from various agents:\n\n"
                for result in collected_results:
                    summary_content += f"[{result['agent']}]: {result['content']}\n\n"

                print("Summary Content:", summary_content)

                summary_input = [
                    ("system", QUESTION_ANSWERING_SYS_PROMPT),
                    ("user", summary_content),
                ]

                res = await question_answering_agent.ainvoke(summary_input)
                final_answer = res.content
                print("Final Answer:", final_answer)

                yield f"data: {json.dumps({'answer': final_answer}, ensure_ascii=False)}\n\n"

            sources = ss.get_sources()
            if sources:
                yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
                await semantic_cache.add(sources=sources)
                ss.clear_sources()
            yield f"data: {json.dumps({'content': '[DONE]'}, ensure_ascii=False)}\n\n"
        except Exception:
            traceback.print_exc()
            res_error = "Sorry, something went wrong while processing your request. Please try again later."
            yield f"data: {json.dumps({'answer': res_error}, ensure_ascii=False)}\n\n"

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
    if not input_data or input_data.strip() == "":
        suggestion = await suggestion_agent.get_welcome_suggestion()
    else:
        suggestion = await suggestion_agent.get_suggestion(question=input_data)
    return suggestion


@app.get("/health")
async def health_check() -> dict:
    """Check the health status of the service.

    Returns:
        dict: A dictionary containing the health status.
    """
    return {"status": "healthy"}


# Autocomplete API endpoints


@app.post("/autocomplete/suggest")
async def autocomplete_suggest(query: AutocompleteQueryModel) -> dict:
    """
    Get autocomplete suggestions for a given prefix.

    Args:
        query (AutocompleteQueryModel): Contains prefix and max_suggestions

    Returns:
        dict: List of suggestions with word and frequency
    """
    try:
        suggestions = autocomplete_trie.get_suggestions(
            prefix=query.prefix, max_suggestions=query.max_suggestions
        )
        return {
            "suggestions": suggestions,
            "prefix": query.prefix,
            "count": len(suggestions),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting suggestions: {str(e)}"
        )


@app.post("/autocomplete/update")
async def autocomplete_update_frequency(update: AutocompleteUpdateModel) -> dict:
    """
    Update the frequency of a word in the trie (e.g., when user selects a suggestion).

    Args:
        update (AutocompleteUpdateModel): Contains word and increment value

    Returns:
        dict: Success message
    """
    try:
        autocomplete_trie.update_frequency(word=update.word, increment=update.increment)
        autocomplete_trie.save_to_disk()  # Persist the update
        return {
            "message": f"Updated frequency for '{update.word}' by {update.increment}",
            "word": update.word,
            "increment": update.increment,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating frequency: {str(e)}"
        )


@app.post("/autocomplete/load")
async def autocomplete_load_data(load_request: AutocompleteLoadModel) -> dict:
    """
    Load search queries from a text file into the trie.

    Args:
        load_request (AutocompleteLoadModel): Contains file path

    Returns:
        dict: Load status and statistics
    """
    try:
        autocomplete_trie.load_from_text_file(load_request.file_path)
        autocomplete_trie.save_to_disk()  # Persist the loaded data
        stats = autocomplete_trie.get_stats()
        return {
            "message": f"Successfully loaded data from {load_request.file_path}",
            "stats": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")


@app.get("/autocomplete/stats")
async def autocomplete_stats() -> dict:
    """
    Get statistics about the autocomplete trie.

    Returns:
        dict: Trie statistics including word count, cache info, etc.
    """
    try:
        stats = autocomplete_trie.get_stats()
        top_queries = autocomplete_trie.get_top_queries(limit=10)
        return {"stats": stats, "top_queries": top_queries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/autocomplete/save")
async def autocomplete_save() -> dict:
    """
    Manually save the trie to disk.

    Returns:
        dict: Save status
    """
    try:
        autocomplete_trie.save_to_disk()
        return {"message": "Trie successfully saved to disk"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving trie: {str(e)}")


@app.post("/autocomplete/clear-cache")
async def autocomplete_clear_cache() -> dict:
    """
    Clear the LRU caches to free memory or force cache refresh.

    Returns:
        dict: Cache clear status
    """
    try:
        autocomplete_trie.clear_cache()
        return {"message": "LRU caches cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")
