from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models
from langgraph.graph import END

# import sys

# sys.path.append(
#     "/Users/lihaozhe/Coding/omni/core"
# )  # Adjust the path to import from core
from core.agents.research import research_agent
from core.agents.math import math_agent
from core.agents.web_browsing import web_page_agent
from core.agents.timing import timing_agent
from core.agents.coding import coding_agent
from core.agents.summarizing import summarizing_agent
from core.agents.weather import weather_agent


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        # highlight-next-line
        return Command(
            # highlight-next-line
            goto=agent_name,  # (1)!
            # highlight-next-line
            update={**state, "messages": state["messages"] + [tool_message]},  # (2)!
            # highlight-next-line
            graph=Command.PARENT,  # (3)!
        )

    return handoff_tool


# Handoffs
assign_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent.",
)

assign_to_math_agent = create_handoff_tool(
    agent_name="math_agent",
    description="Assign task to a math agent.",
)

assign_to_web_page_agent = create_handoff_tool(
    agent_name="web_page_agent",
    description="Assign task to a web page agent.",
)

assign_to_planning_agent = create_handoff_tool(
    agent_name="planning_agent",
    description="Assign task to a planning agent.",
)

assign_to_timing_agent = create_handoff_tool(
    agent_name="timing_agent",
    description="Assign task to a timing agent.",
)

assign_to_coding_agent = create_handoff_tool(
    agent_name="coding_agent",
    description="Assign task to a coding agent.",
)

assign_to_summarizing_agent = create_handoff_tool(
    agent_name="summarizing_agent",
    description="Assign task to a summarizing agent.",
)

assign_to_weather_agent = create_handoff_tool(
    agent_name="weather_agent",
    description="Assign task to a weather agent.",
)
tools = [
    assign_to_research_agent,
    assign_to_math_agent,
    assign_to_web_page_agent,
    assign_to_timing_agent,
    assign_to_coding_agent,
    assign_to_summarizing_agent,
    assign_to_weather_agent,
]
supervisor_agent = create_react_agent(
    model=default_llm_models.supervisor_model,
    tools=tools,
    prompt=(
        "RESEARCH SUPERVISOR - CONDUCT COMPREHENSIVE INVESTIGATION\n\n"
        "You are a research supervisor managing a team of specialized agents. Your job is to conduct thorough research by strategically delegating tasks to the appropriate agents.\n\n"
        "AVAILABLE SPECIALIST AGENTS:\n"
        "- **Research Agent**: Browse current information over the internet, gather facts and data\n"
        "- **Math Agent**: Perform mathematical calculations, analysis, and problem-solving\n"
        "- **Web Page Agent**: ONLY use when user explicitly provides a webpage URL to analyze - DO NOT call proactively\n"
        "- **Timing Agent**: Provide exact datetime information, handle time-related queries\n"
        "- **Coding Agent**: Write, execute, and debug code with output analysis\n"
        "- **Weather Agent**: Provide current weather information for specific locations\n"
        "- **Summarizing Agent**: FINAL STEP - Synthesize all research findings into comprehensive answer\n\n"
        "BUDGET CONSTRAINT - MAXIMUM 8 AGENT CALLS:\n"
        "Count each agent delegation as 1 call. Examples:\n"
        "- research_agent → coding_agent → math_agent → research_agent → summarizing_agent = 5 calls ✓\n"
        "- timing_agent → research_agent → summarizing_agent = 3 calls ✓\n"
        "**Stay within 8 calls total including the final summarizing_agent call**\n\n"
        "RESEARCH STRATEGY:\n"
        "1. **Analyze the Question**: What specific information is needed? What are the key components?\n"
        "2. **Plan Your Investigation**: Break down complex questions into research phases (max 8 steps)\n"
        "3. **Delegate Strategically**: Choose the right agents for each research task\n"
        "4. **Iterative Research**: Based on findings, you may need to:\n"
        "   - Call the same agent again with refined questions\n"
        "   - Call different agents to explore new directions\n"
        "   - Gather additional context before proceeding\n"
        "5. **Research Completion**: Only call summarizing agent when you have sufficient information\n\n"
        "AGENT FAILURE & RECOVERY PROTOCOLS:\n"
        "- **If an agent fails**: Try once more, then switch to alternative agent\n"
        "- **Weather Agent fails**: Use research_agent to search for weather information\n"
        "- **Research Agent poor results**: May be due to cache - specify 'no cache' on retry\n"
        "- **Web Page Agent**: Only use when user explicitly provides URLs - never call proactively\n\n"
        "RESEARCH WORKFLOW EXAMPLES (within budget):\n"
        "- Simple query: research_agent → summarizing_agent (2 calls)\n"
        "- Time-sensitive: timing_agent → research_agent → summarizing_agent (3 calls)\n"
        "- Calculation-heavy: research_agent → math_agent → research_agent → summarizing_agent (4 calls)\n"
        "- Multi-faceted: research_agent → coding_agent → math_agent → research_agent → summarizing_agent (5 calls)\n"
        "- Complex research: research_agent → coding_agent → research_agent → math_agent → research_agent → summarizing_agent (6 calls)\n\n"
        "CRITICAL RULES:\n"
        "- **NEVER answer questions directly** - You are a supervisor, not a direct answerer\n"
        "- **BUDGET AWARENESS** - Never exceed 8 agent calls total\n"
        "- **Web Page Agent Restriction** - Only use when user explicitly provides webpage URLs\n"
        "- **Agent Failure Protocol** - Retry once, then use alternative agent\n"
        "- **Research Agent Cache** - If poor results, specify 'no cache' on retry\n"
        "- **Strategic delegation** - Choose agents based on their specialized capabilities\n"
        "- **Final synthesis** - Always end with summarizing_agent to provide the comprehensive answer\n"
        "- **Efficiency over perfection** - Work within budget constraints\n\n"
        "DECISION FRAMEWORK:\n"
        "Before each delegation ask:\n"
        "- What specific information do I need?\n"
        "- Which agent is best suited for this task?\n"
        "- How many calls have I made? (Stay under 8)\n"
        "- Do I have enough context for the agent to work effectively?\n"
        "- After receiving results: Is this sufficient or do I need more investigation?\n"
        "- If agent failed: Should I retry once or switch to alternative?\n\n"
    ),
    name="supervisor",
)

# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(
        supervisor_agent,
        destinations=(
            "research_agent",
            "math_agent",
            "web_page_agent",
            "timing_agent",
            "coding_agent",
            "summarizing_agent",
            "weather_agent",
        ),
    )
    .add_node(research_agent)
    .add_node(math_agent)
    .add_node(web_page_agent)
    .add_node(timing_agent)
    .add_node(coding_agent)
    .add_node(summarizing_agent)
    .add_node(weather_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .add_edge("web_page_agent", "supervisor")
    .add_edge("timing_agent", "supervisor")
    .add_edge("coding_agent", "supervisor")
    .add_edge("weather_agent", "supervisor")
    .add_edge("summarizing_agent", END)
    .compile()
)
