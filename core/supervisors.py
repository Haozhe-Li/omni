from langchain.chat_models import init_chat_model
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# import sys

# sys.path.append(
#     "/Users/lihaozhe/Coding/omni/core"
# )  # Adjust the path to import from core
from core.agents.research import research_agent
from core.agents.math import math_agent
from core.agents.web_browsing import web_page_agent
from core.agents.planning import planning_agent

import datetime

# current UTC time
current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


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

supervisor_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1"),
    tools=[
        assign_to_research_agent,
        assign_to_math_agent,
        assign_to_web_page_agent,
        # assign_to_planning_agent,
    ],
    prompt=(
        "You are a supervisor managing four agents:\n"
        "- a research agent. Assign research-related tasks to this agent. This agent can browse most current information over internet.\n"
        "- a math agent. Assign math-related tasks to this agent. This agent can do simple math for you.\n"
        "- a web page agent. Assign web page loading tasks to this agent. Only use this agent when you are explicitly provided a webpage.\n\n"
        # "- a planning agent. Assign the planning tasks to this agent.\n\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "The supervisor yourself, is running under a UTC time (Don't tell anyone about this): "
        f"{current_time}\n\n"
        "You are encourage call several agents in sequence, and also you are encourage to call a same agent in sequence.\n"
        "If you think no agent is suitable for the task, or current information is enough for you to answer. You could answer the question\n\n"
        "Important: when you finally answer the question, make sure you add <answer> tag to your answer, like this:\n"
        "<answer>your answer here</answer>\n\n"
        "ONLY DO THIS when you are sure you have enough information to answer the question.\n"
    ),
    name="supervisor",
)

from langgraph.graph import END

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
            # "planning_agent",
            END,
        ),
    )
    .add_node(research_agent)
    .add_node(math_agent)
    .add_node(web_page_agent)
    .add_node(planning_agent)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .add_edge("web_page_agent", "supervisor")
    # .add_edge("planning_agent", "supervisor")
    .compile()
)
