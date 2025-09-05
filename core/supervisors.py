from typing import Annotated, Optional
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models
from langgraph.graph import END
from core.agents.research import research_agent
from core.agents.math import math_agent
from core.agents.web_browsing import web_page_agent
from core.agents.coding import coding_agent
from core.agents.summarizing import summarizing_agent
from core.agents.weather import weather_agent


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """Create a handoff tool that routes control to another agent with optional instruction.

    Added feature: an "instruction" parameter that the supervisor must supply to
    clarify the exact subtask focus, expected angle, constraints, or desired output
    format for the target agent. This encourages deliberate decomposition.
    """

    name = f"transfer_to_{agent_name}"
    description = (
        description
        or f"Delegate a subtask to {agent_name}. Always include an 'instruction' parameter specifying the precise focus."
    )

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        instruction: Optional[str] = None,
    ) -> Command:
        """Handoff execution.

        Args:
            state: Current graph message state (injected).
            tool_call_id: Tool call identifier (injected).
            instruction: A concise directive for the target agent. Should state:
                - Subtask objective / angle
                - Key context / constraints
                - Expected output form (e.g., list of facts, python code, numeric proof)
        """

        tool_message = {
            "role": "tool",
            "content": f"Transferred to {agent_name}. Instruction attached: {bool(instruction)}",
            "name": name,
            "tool_call_id": tool_call_id,
        }

        new_messages = state["messages"] + [tool_message]
        if agent_name == "summarizing_agent":
            instruction = "Please provide a well-structured and comprehensive answer."
        if instruction:
            new_messages.append(
                {
                    "role": "user",
                    "content": f"<delegation_instruction target='{agent_name}'>\n{instruction.strip()}\n</delegation_instruction>",
                }
            )

        return Command(
            goto=agent_name,
            update={**state, "messages": new_messages},
            graph=Command.PARENT,
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
    assign_to_coding_agent,
    assign_to_summarizing_agent,
    assign_to_weather_agent,
]
supervisor_agent = create_react_agent(
    model=default_llm_models.supervisor_model,
    tools=tools,
    prompt=(
        "OMNI SUPERVISOR — Planner & Delegator\n\n"
        "Goal: Solve the user's request with minimal-cost multi-hop reasoning. Delegate exactly one focused subtask per step.\n"
        "MANDATORY: You must ALWAYS end by calling summarizing_agent. Never answer the user directly yourself.\n\n"
        "Agents:\n"
        "- research_agent — web search (exactly ONE search per call).\n"
        "- web_page_agent — open a specific URL and extract a targeted answer.\n"
        "- math_agent — formal math/logic (do not use coding_agent for pure math).\n"
        "- coding_agent — programming or code execution tasks.\n"
        "- weather_agent — current weather.\n"
        "- summarizing_agent — final synthesis (FREE, mandatory last call).\n\n"
        "Budget: Max 8 paid calls (excluding summarizing_agent). If you reach or will reach 8 on the next step, call summarizing_agent immediately.\n\n"
        "Loop per step:\n"
        "1) State the current objective.\n"
        "2) Pick the single best agent for the smallest verifiable next sub-question.\n"
        "3) Call transfer_to_<agent> with { instruction: '...' }.\n"
        "4) After the agent returns, add <reflection>...</reflection> on what you learned and what's missing.\n"
        "5) If more is needed, add a one-step <plan>...</plan> and continue; otherwise call summarizing_agent.\n\n"
        "Instruction (required for every transfer):\n"
        "- Objective: exact sub-question\n"
        "- Scope: constraints to stay focused\n"
        "- Output: expected form (e.g., 3–5 bullet facts, number, code)\n"
        'Example: instruction="Find NVIDIA data center revenue YoY growth for the last 6 months."\n\n'
        "web_page_agent instruction must include:\n"
        "- Exact URL (e.g., https://example.com)\n"
        "- The specific item to verify/extract\n"
        'Example: instruction="Open https://tesla.com and confirm whether 2024 revenue > $90B."\n\n'
        "Rules:\n"
        "- Use tools only; do not fabricate.\n"
        "- Retry once on failure; if still blocked, switch strategy or acknowledge limits, then move to summarizing_agent.\n"
        "- Track call count; never exceed budget.\n"
        "- If the user asks about Omni itself, go directly to summarizing_agent.\n\n"
        "Stop: When core sub-questions are answered or marginal value is low, call summarizing_agent to produce the final answer."
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
            "coding_agent",
            "summarizing_agent",
            "weather_agent",
        ),
    )
    .add_node(research_agent)
    .add_node(math_agent)
    .add_node(web_page_agent)
    .add_node(coding_agent)
    .add_node(summarizing_agent)
    .add_node(weather_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .add_edge("web_page_agent", "supervisor")
    .add_edge("coding_agent", "supervisor")
    .add_edge("weather_agent", "supervisor")
    .add_edge("summarizing_agent", END)
    .compile()
)
