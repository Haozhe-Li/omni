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

assign_to_weather_agent = create_handoff_tool(
    agent_name="weather_agent",
    description="Assign task to a weather agent.",
)
tools = [
    assign_to_research_agent,
    assign_to_math_agent,
    assign_to_web_page_agent,
    assign_to_coding_agent,
    assign_to_weather_agent,
]

supervisor_agent = create_react_agent(
    model=default_llm_models.supervisor_model,
    tools=tools,
    prompt=(
        "You are an intelligent task coordinator with a bias toward comprehensive information gathering. Your role is to analyze user requests and strategically delegate tasks to specialized agents to provide the most thorough and accurate responses possible.\n\n"
        "Core Philosophy: Never underestimate any question. Even seemingly simple queries can benefit from deeper investigation. Always err on the side of gathering more information rather than less. Multiple perspectives and data sources lead to better outcomes.\n\n"
        "Available Agents:\n"
        "- research_agent: Web search and information gathering from multiple sources\n"
        "- web_page_agent: Extract information from specific URLs (ONLY use URLs explicitly provided by the user)\n"
        "- math_agent: Mathematical calculations, logical reasoning, and quantitative analysis\n"
        "- coding_agent: Programming, code execution, technical implementation, and debugging\n"
        "- weather_agent: Current weather information and meteorological data\n\n"
        "Strategic Instructions:\n"
        "1. Break down complex requests into multiple subtasks and delegate each to the most appropriate agent\n"
        "2. Consider calling multiple agents for different perspectives on the same topic\n"
        "3. Gather comprehensive background information before making final recommendations\n"
        "4. Use agents to verify and cross-reference information from different sources\n"
        "5. You may delegate up to 10 agent calls maximum per user request\n\n"
        "Critical Constraints:\n"
        "- NEVER use web_page_agent with fabricated or assumed URLs\n"
        "- ONLY use web_page_agent when the user explicitly provides a specific URL\n"
        "- When in doubt about which agent to use, delegate to multiple agents for broader coverage\n\n"
        "Delegation Guidelines:\n"
        "- Provide detailed, specific instructions to each agent\n"
        "- Include all relevant context and constraints\n"
        "- Specify the expected format and depth of response\n"
        "- Consider the interconnections between different aspects of the user's request\n\n"
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
            "weather_agent",
        ),
    )
    .add_node(research_agent)
    .add_node(math_agent)
    .add_node(web_page_agent)
    .add_node(coding_agent)
    .add_node(weather_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .add_edge("web_page_agent", "supervisor")
    .add_edge("coding_agent", "supervisor")
    .add_edge("weather_agent", "supervisor")
    .add_edge("supervisor", END)
    .compile()
)
