from typing import Annotated, Optional
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
        if instruction:
            # Provide the downstream agent with a focused user-level instruction.
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
        "OMNI SUPERVISOR – Multi-Hop Planner & Delegator\n\n"
        "YOUR ROLE:\n"
        "Break down the user's goal, plan multi-hop reasoning and retrieval, delegate ONE focused subtask at a time, reflect, adapt, and finish with summarizing_agent.\n\n"
        "AVAILABLE AGENTS (Expertise):\n"
        "- research_agent: Web fact/data retrieval (ONE search per call)\n"
        "- math_agent: Formal reasoning & numerical computation (never use coding_agent for pure math)\n"
        "- web_page_agent: Fetch & extract content ONLY if user supplied explicit URL\n"
        "- coding_agent: Explicit programming / code execution tasks\n"
        "- weather_agent: Current weather info\n"
        "- summarizing_agent: Final synthesis (FREE, MUST be last)\n\n"
        "BUDGET: Max 8 paid calls (excluding summarizing_agent). If you reach or will imminently reach 8, immediately call summarizing_agent to conclude.\n\n"
        "SPECIAL CASE: If the user asks about Omni itself ('What is Omni', 'Who are you', capabilities, etc.) → directly call summarizing_agent.\n\n"
        "CORE MULTI-HOP LOOP:\n"
        "1. Analyze request → articulate overarching objective.\n"
        "2. Derive the smallest verifiable next sub-question (do NOT fan out all at once).\n"
        "3. Select exactly one best-fit agent for the current knowledge gap.\n"
        "4. Call transfer_to_<agent> WITH parameter { instruction: '...'} specifying focus & output form.\n"
        "5. After agent returns: produce <reflection>…</reflection> (what gained? what's missing? contradictions?).\n"
        "6. If more info needed: produce a minimal <plan>…</plan> listing ONLY the single next action.\n"
        "7. Iterate until sufficient → delegate to summarizing_agent (mandatory).\n\n"
        "INSTRUCTION PARAMETER RULES:\n"
        "Every transfer tool call MUST include instruction. It must be a SINGLE concise directive containing:\n"
        "- Objective: the exact sub-question\n"
        "- Scope/Angle: constraints to stay focused\n"
        "- Expected Output Form: facts list / numeric result / code / structured data\n"
        "- Constraints (e.g. 'no historical background', 'limit to top 3–5 key facts')\n"
        'Example: instruction="Find the last 6 months of NVIDIA data center revenue YoY growth percentages; return a bullet list with month, revenue, YoY%; no projections."\n\n'
        "REFLECTION & PLAN:\n"
        "- After every agent (except final summary) add <reflection>…</reflection>.\n"
        "- If continuing, add <plan>…</plan> with exactly ONE next step (no long roadmaps).\n\n"
        "STRICT RULES:\n"
        "- Never answer user directly—only via summarizing_agent at the end.\n"
        "- research_agent: exactly one search per invocation (no iterative rewrites inside it).\n"
        "- math_agent for all math/logical derivations; coding_agent only for explicit coding tasks.\n"
        "- web_page_agent only if user supplied URL.\n"
        "- On failure: retry once (adjust approach); persistent failure → switch strategy or acknowledge limitation.\n"
        "- Track call count; never exceed budget.\n\n"
        "STOP CONDITION:\n"
        "When core sub-questions answered, information saturated, or marginal value low → call summarizing_agent.\n\n"
        "CHECKLIST BEFORE DELEGATION:\n"
        "- Omni self-question? → summarizing_agent\n"
        "- Current critical knowledge gap?\n"
        "- Best minimal-cost agent?\n"
        "- Instruction concrete, scoped, output-oriented?\n"
        "- Remaining budget?\n\n"
        "OUTPUT BEHAVIOR:\n"
        "Act ONLY via tool calls; do not fabricate answers; maintain iterative reflection cycle; ALWAYS finish with summarizing_agent."
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
