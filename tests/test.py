# %% [markdown]
# # Agentic RAG
#
# In this tutorial we will build a [retrieval agent](https://python.langchain.com/docs/tutorials/qa_chat_history). Retrieval agents are useful when you want an LLM to make a decision about whether to retrieve context from a vectorstore or respond to the user directly.
#
# By the end of the tutorial we will have done the following:
#
# 1. Fetch and preprocess documents that will be used for retrieval.
# 2. Index those documents for semantic search and create a retriever tool for the agent.
# 3. Build an agentic RAG system that can decide when to use the retriever tool.
#
# ![Screenshot 2024-02-14 at 3.43.58 PM.png](attachment:7ad1a116-28d7-473f-8cff-5f2efd0bf118.png)
#
# ## Setup
#
# Let's download the required packages and set our API keys:

# %%
# %%capture --no-stderr
# %pip install -U --quiet langgraph "langchain[openai]" langchain-community langchain-text-splitters

# %% [markdown]
# <div class="admonition tip">
#     <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
#     <p style="padding-top: 5px;">
#         Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>.
#     </p>
# </div>

# %% [markdown]
# ## 1. Preprocess documents
#
# 1\. Fetch documents to use in our RAG system. We will use three of the most recent pages from [Lilian Weng's excellent blog](https://lilianweng.github.io/). We'll start by fetching the content of the pages using `WebBaseLoader` utility:

# %% [markdown]
# 2\. Split the fetched documents into smaller chunks for indexing into our vectorstore:

# %% [markdown]
# ## 2. Create a retriever tool

# %% [markdown]
# Now that we have our split documents, we can index them into a vector store that we'll use for semantic search.
#
# 1\. Use an in-memory vector store and OpenAI embeddings:

# %% [markdown]
# 2\. Create a retriever tool using LangChain's prebuilt `create_retriever_tool`:

# %% [markdown]
# 3\. Test the tool:

# %%
# a python excuator allow LLM to write and execute python code
from langchain_core.tools import Tool
import sys
from io import StringIO
from langchain_core.tools import tool
from langchain_community.tools.riza.command import ExecPython


from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool


from langchain_core.tools import Tool


# Fix the load_webpage tool with a proper docstring
@tool
def load_webpage(url: str) -> str:
    """Load a webpage and return its content.

    Args:
        url: The URL of the webpage to load.

    Returns:
        The content of the webpage as a string.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    # Combine all document content into a single string
    return "\n".join(doc.page_content for doc in docs)


# Fix the search_tool with a proper docstring
search = DuckDuckGoSearchRun()


@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo and return the results.

    Args:
        query: The search query string.

    Returns:
        The search results as a string.
    """
    return search.run({"query": query})


# %%
# python_tool_wrapper("print('Hello, world!')\n")

# %% [markdown]
# ## 3. Generate query

# %% [markdown]
# Now we will start building components ([nodes](../../../concepts/low_level#nodes) and [edges](../../../concepts/low_level#edges)) for our agentic RAG graph. Note that the components will operate on the [`MessagesState`](../../../concepts/low_level#messagesstate) — graph state that contains a `messages` key with a list of [chat messages](https://python.langchain.com/docs/concepts/messages/).
#
# 1\. Build a `generate_query_or_respond` node. It will call an LLM to generate a response based on the current graph state (list of messages). Given the input messages, it will decide to retrieve using the retriever tool, or respond directly to the user. Note that we're giving the chat model access to the `retriever_tool` we created earlier via `.bind_tools`:

# %%
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model("openai:gpt-4.1-nano", temperature=0)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        # highlight-next-line
        .bind_tools([search_tool, load_webpage]).invoke(state["messages"])
    )
    return {"messages": [response]}


# %% [markdown]
# 2\. Try it on a random input:

# %%
input = {"messages": [{"role": "user", "content": "3 * 4"}]}
generate_query_or_respond(input)["messages"][-1].pretty_print()

# %% [markdown]
# 3\. Ask a question that requires semantic search:

# %%
input = {
    "messages": [
        {
            "role": "user",
            "content": "What does Lilian Weng say about types of reward hacking?",
        }
    ]
}
generate_query_or_respond(input)["messages"][-1].pretty_print()

# %% [markdown]
# ## 4. Grade documents

# %% [markdown]
# 1\. Add a [conditional edge](../../../concepts/low_level#conditional-edges) — `grade_documents` — to determine whether the retrieved documents are relevant to the question. We will use a model with a structured output schema `GradeDocuments` for document grading. The `grade_documents` function will return the name of the node to go to based on the grading decision (`generate_answer` or `rewrite_question`):

# %%
from pydantic import BaseModel, Field
from typing import Literal

EVAL_PROMPT = (
    "You are a evaluator assessing relevance of a retrieved document to a user question. \n "
    "Here's the budget left {budget} / 5. Budget here means the remaining number of no you can respond. \n"
    "When the budget is higher than 2, you are expected to be more strict in your evaluation. That means you usually come up with no in your first response.\n"
    "Your job is to determine whether the retrived document provides enough information to answer the question.\n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the information is enough. \n"
)


# highlight-next-line
class EvaluateDocuments(BaseModel):
    """Evaluate documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


eval_model = init_chat_model("openai:gpt-4.1-nano", temperature=0)


class EvaluateBudget:
    def __init__(self):
        self.budget = 5

    def use_budget(self):
        """Use one budget unit."""
        if self.budget > 0:
            self.budget -= 1
            return self.budget
        else:
            return 0

    def get_budget(self):
        """Get the current budget."""
        return self.budget

    def reset_budget(self):
        """Reset the budget to the initial value."""
        self.budget = 5


evalBudget = EvaluateBudget()


def evaluate_information(
    state: MessagesState,
) -> Literal["generate_answer", "continue_research"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    # print("Remaining budget:", evalBudget.get_budget())

    prompt = EVAL_PROMPT.format(
        question=question, context=context, budget=evalBudget.get_budget()
    )
    print(prompt)
    response = (
        eval_model
        # highlight-next-line
        .with_structured_output(EvaluateDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes" or evalBudget.get_budget() <= 0:
        evalBudget.reset_budget()
        print("Choose yes, budget reset")
        return "generate_answer"
    else:
        evalBudget.use_budget()
        print("Continuing research, budget left:", evalBudget.get_budget())
        return "continue_research"


# %% [markdown]
# 2\. Run this with irrelevant documents in the tool response:

# %%
from langchain_core.messages import convert_to_messages

# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {"role": "tool", "content": "meow", "tool_call_id": "1"},
#         ]
#     )
# }
# evaluate_information(input)

# %% [markdown]
# 3\. Confirm that the relevant documents are classified as such:

# %%
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}
evaluate_information(input)

# %% [markdown]
# ## 5. Rewrite question

# %% [markdown]
# 1\. Build the `rewrite_question` node. The retriever tool can return potentially irrelevant documents, which indicates a need to improve the original user question. To do so, we will call the `rewrite_question` node:

# %%

CONTINUE_RESEARCH_PROMPT = (
    "Look at the input and try to reason deeply into this questions.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Now, your job is to come up with a new question that can help you find more information to answer the original question.\n"
    "The new question should be specific and focused, and it should be something that you can search for or retrieve information about.\n"
    "It is a advice that you should break down the original question into smaller parts or aspects that can be addressed individually.\n"
    "Now, formulate a follow up question: "
)


def continue_research(state: MessagesState):
    """Generate a follow-up question to continue research based on the original question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = CONTINUE_RESEARCH_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


# %% [markdown]
# 2\. Try it out:

# %%
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}

response = continue_research(input)
print(response["messages"][-1]["content"])

# %% [markdown]
# ## 6. Generate an answer

# %% [markdown]
# 1\. Build `generate_answer` node: if we pass the grader checks, we can generate the final answer based on the original question and the retrieved context:

# %%
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know.\n"
    "You should answer the question in a concise manner but deeply in detail.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# %% [markdown]
# 2\. Try it:

# %%
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}

response = generate_answer(input)
response["messages"][-1].pretty_print()

# %% [markdown]
# ## 7. Assemble the graph
#
# * Start with a `generate_query_or_respond` and determine if we need to call `retriever_tool`
# * Route to next step using `tools_condition`:
#     * If `generate_query_or_respond` returned `tool_calls`, call `retriever_tool` to retrieve context
#     * Otherwise, respond directly to the user
# * Grade retrieved document content for relevance to the question (`grade_documents`) and route to next step:
#     * If not relevant, rewrite the question using `rewrite_question` and then call `generate_query_or_respond` again
#     * If relevant, proceed to `generate_answer` and generate final response using the `ToolMessage` with the retrieved document context

# %%
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node(
    "retrieve",
    ToolNode([search_tool, load_webpage]),
)
workflow.add_node(continue_research)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    evaluate_information,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("continue_research", "generate_query_or_respond")

# Compile
graph = workflow.compile()

# %% [markdown]
# Visualize the graph:

# %%
# from IPython.display import Image, display

# display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## 8. Run the agentic RAG

# %%
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "explain chatgpt in detail",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        print(update["messages"][-1])
        print("\n\n")
