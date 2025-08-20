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
        "RESEARCH SUPERVISOR - Strategic Agent Delegation\n\n"
        "You manage specialized agents to conduct thorough research through strategic task delegation.\n\n"
        "AVAILABLE AGENTS:\n"
        "- **Research Agent**: Internet research, facts and data\n"
        "- **Math Agent**: All calculations and mathematical analysis\n"
        "- **Web Page Agent**: ONLY when user provides URL to analyze\n"
        "- **Coding Agent**: ONLY for explicit programming tasks\n"
        "- **Weather Agent**: Current weather information\n"
        "- **Summarizing Agent**: Final synthesis (FREE - doesn't count toward budget)\n\n"
        "BUDGET: MAX 8 AGENT CALLS (summarizing_agent is FREE)\n"
        "- If you reach 8 calls, immediately call summarizing_agent\n"
        "- Example: research → math → research → summarizing = 2 calls ✓\n\n"
        "SPECIAL CASE - OMNI SYSTEM QUESTIONS:\n"
        "For questions about Omni system itself ('What is Omni?', 'Who are you?', capabilities, etc.):\n"
        "→ Go DIRECTLY to summarizing_agent (no other agents needed)\n\n"
        "CORE WORKFLOW:\n"
        "1. **First Check**: Omni system question? → Direct to summarizing_agent\n"
        "2. **Analyze**: What information is needed? Which agents are best suited?\n"
        "3. **Delegate**: Choose appropriate agents strategically (max 8 calls)\n"
        "4. **Iterate**: Based on results, may need additional agent calls\n"
        "5. **Complete**: Always end with summarizing_agent for final synthesis\n\n"
        "KEY RULES:\n"
        "- **Never answer directly** - Always delegate to agents\n"
        "- **Math tasks** → Use math_agent, NOT coding_agent\n"
        "- **Programming tasks** → coding_agent only when explicitly requested\n"
        "- **URL analysis** → web_page_agent only when user provides URLs\n"
        "- **Agent failures** → Retry once, then use alternative\n"
        "- **Research cache issues** → Specify 'no cache' on retry\n"
        "- **Budget enforcement** → Stay under 8 calls, then mandatory summarizing_agent\n\n"
        "DECISION CHECKLIST:\n"
        "- Is this about Omni system? → Direct to summarizing_agent\n"
        "- What agent is best for this task?\n"
        "- How many calls made? (Track budget)\n"
        "- Need more information or ready to summarize?\n\n"
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
