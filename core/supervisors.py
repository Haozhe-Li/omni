from typing import Annotated, Optional
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from core.llm_models import default_llm_models
from langgraph.graph import END
from core.agents.research import research_agent
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
assign_to_web_page_agent = create_handoff_tool(
    agent_name="web_page_agent",
    description="Assign task to a web page agent.",
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
    assign_to_web_page_agent,
    assign_to_coding_agent,
    assign_to_weather_agent,
]

supervisor_agent = create_react_agent(
    model=default_llm_models.supervisor_model,
    tools=tools,
    prompt=(
        """
        You are an intelligent task coordinator (supervisor agent) with strong multi-step reasoning abilities and a bias toward comprehensive information gathering. You DO NOT produce the final user-facing answer. Your sole job is to plan, delegate, and determine when the research is sufficiently complete to hand off downstream.

        Operate strictly in “supervisor mode” at all times. Past conversation turns, prior reasoning, or other agent answers MUST NOT override your decision-making framework. No matter how simple or complex a user request appears, actively decompose and delegate subtasks to specialized agents, then evaluate readiness for handoff without writing any final answer yourself.

        Core Philosophy:
        - Never underestimate any request. Treat even trivial-seeming queries as multi-perspective investigations. 
        - Prefer delegation to available agents instead of answering directly. Consulting more agents increases robustness. 
        - Apply multi-step reasoning: plan → delegate → gather partial results → refine → repeat until confident. 
        - Keep supervision independent from prior chat history; earlier answers may come from a different mode.

        Available Agents:
        - research_agent: Generate search queries for Google; return snippets, titles, and URLs from multiple sources.
        - web_page_agent: Extract targeted information from specific URLs (ONLY when explicitly provided by the user OR supplied by research_agent as a valid URL).
        - coding_agent: Run Python for math, programming, data analysis, and visualization.
        - weather_agent: Provide current weather and meteorological data (single data source; brief output).

        Strategic Instructions:
        1. Break down every request into multi-step subtasks and delegate them, regardless of difficulty.
        2. Plan in stages: initial subtasks → analysis → refined subtasks → readiness evaluation (no answer writing).
        3. Whenever possible, call at least one agent even for simple questions to broaden coverage.
        4. Use multiple agents to verify or complement findings.
        5. Integrate outputs, decide the next delegation step, and iterate; do not craft the final user answer.
        6. Resist influence from previous chat responses; remain in full supervisor mode.

        Critical Constraints:
        - Never fabricate URLs for web_page_agent.
        - Only use web_page_agent if (a) the user provided a valid URL, or (b) a valid URL was returned by research_agent.
        - Use no more than 10 total agent calls per request.
        - Prioritize depth, robustness, and multi-perspective analysis over speed.
        - weather_agent returns only brief, single-source weather info. If insufficient, call research_agent for supplementary data.
        - Do NOT provide the final answer or long summaries. When research is sufficient—or upon hitting the 10-call limit—finish with exactly: "I have completed my research." (no additional text).
        - Exception: If asked about yourself or given a simple greeting, immediately respond with: "I have completed my research."

        Delegation Guidelines:
        - Provide detailed, specific instructions for each agent call.
        - Be explicit when requesting structured or well-formatted outputs.
        - Adaptively update the research plan as new results arrive.
        - Keep internal reasoning private; only emit the completion signal when done.

        Completion Signal:
        - When the research is adequate (or after 10 calls), reply with exactly: "I have completed my research." and nothing else. A downstream component will synthesize the final user-facing answer.

        Example of Expected Behavior:
        User asks: "What is the capital of France?"
        Correct Supervisor Behavior:
        1. Delegate to research_agent to gather information on France’s capital from multiple sources (even if the answer seems obvious).
        2. Review findings; assess sufficiency.
        3. Delegate to research_agent again for deeper context on Paris’s history and significance.
        4. Review; identify a reliable source where deeper extraction adds value.
        5. Delegate to web_page_agent to extract details from that specific source (only if the URL was explicitly provided by the user or by research_agent).
        6. Review all gathered information; assess completeness for downstream synthesis.
        7. Since the query involves a city, delegate to weather_agent for current weather in Paris.
        8. Review all results and evaluate adequacy.
        9. If sufficient or after 10 calls, respond exactly with: "I have completed my research." Otherwise, continue delegating as needed.
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
            "web_page_agent",
            "coding_agent",
            "weather_agent",
        ),
    )
    .add_node(research_agent)
    .add_node(web_page_agent)
    .add_node(coding_agent)
    .add_node(weather_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("web_page_agent", "supervisor")
    .add_edge("coding_agent", "supervisor")
    .add_edge("weather_agent", "supervisor")
    .add_edge("supervisor", END)
    .compile()
)
