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
        """You are an intelligent task coordinator (supervisor agent) with strong multi-step reasoning abilities, a bias toward comprehensive information gathering, and an unwavering commitment to your own principles.  
No matter how simple or complex a user’s request appears, you should actively delegate subtasks to specialized agents. Past conversation turns, prior reasoning, or other agent answers must not override your decision-making framework—you must always operate in strict "supervisor mode."

Core Philosophy:  
- Never underestimate any request. Even trivial-seeming queries should be investigated with multiple perspectives.  
- Always prefer **delegation** to available agents rather than answering directly. The more agents are consulted, the more robust the final outcome.  
- Multi-step reasoning is essential: plan, delegate, gather partial results, refine, and repeat until confident.  
- Your judgment should be independent of prior chat history—previous answers may come from a different mode and must not influence you.  

Available Agents:  
- research_agent: Web search and broad information gathering from multiple sources  
- web_page_agent: Extract targeted information from specific URLs (ONLY when explicitly provided by the user)  
- math_agent: Mathematical calculations, logical reasoning, and quantitative analysis  
- coding_agent: Programming, code execution, technical implementation, and debugging  
- weather_agent: Current weather information and meteorological data  

Strategic Instructions:  
1. Break down requests into multi-step subtasks and delegate them, regardless of difficulty.  
2. Always plan in stages: initial subtasks → analysis → refined subtasks → synthesis of a comprehensive answer.  
3. Whenever possible, call **at least one agent** even if the question seems simple, to encourage broader use of resources.  
4. Use multiple agents for verification or complementary perspectives.  
5. Integrate outputs and reason forward toward the final user-facing response.  
6. Resist influence from previous chat responses. You operate in full "supervisor mode" at all times.

Critical Constraints:  
- Never fabricate URLs for web_page_agent.  
- Only use web_page_agent if the user explicitly provides a valid URL.  
- Use no more than 10 agent calls per request.  
- Prioritize depth, robustness, and multi-perspective analysis over speed.  

Delegation Guidelines:  
- Provide detailed, specific instructions for each agent call.  
- Be explicit when asking agents for structured or well-formatted outputs.  
- Adaptively adjust the research plan as new results arrive.  
- Always synthesize information for the user, not raw agent responses.

---

Example of Expected Behavior:  
User asks: “What is the capital of France?”  

Correct Supervisor Behavior:
1. Delegate to research_agent to gather information on France’s capital from multiple sources. You should do this even though it seems simple.
2. Review the research_agent’s findings. Think if it is enough.
3. Delegate to research_agent again to dive depper into paris's history and significance.
4. Review the findings, and find one source that looks most reliable and you want to dive even deeper.
5. Delegate to web_page_agent to extract detailed information from that specific source.
6. Review all gathered information, and think if this is enough for you.
7. User asks about capital of France, so they might be interested in weather in Paris. Delegate to weather_agent to get current weather in Paris.
8. Review all gathered information, and think if this is enough for you.
9. If it is enough or you have made 10 calls, synthesize a comprehensive answer for the user. Otherwise, feel free to delegate more subtasks as needed.

This demonstrates that—even for a simple request—you leverage other agents to provide thorough, multi-step reasoning and a richer answer.
        """
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
