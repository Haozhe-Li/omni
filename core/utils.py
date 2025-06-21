# hide-cell
from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def pretty_yield_messages(update, last_message=False):
    """
    Similar to pretty_print_messages but yields formatted content for streaming.

    Args:
        update: The update from the supervisor
        last_message: Whether to return only the last message

    Yields:
        Formatted message content parts as strings
    """
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        yield f"Update from subgraph {graph_id}:\n\n"
        is_subgraph = True

    for node_name, node_update in update.items():
        # update_label = f"Update from node {node_name}:\n\n"
        # if is_subgraph:
        #     update_label = "\t" + update_label
        # yield update_label

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_message = m.pretty_repr(html=True)
            if is_subgraph:
                pretty_message = "\n".join("\t" + c for c in pretty_message.split("\n"))
            yield pretty_message

        yield "\n"


def clean_messages(messages: str):
    # stipe, remove extra spaces, and newlines
    cleaned = messages.strip()
    import re

    # remove anything between 3!!!
    cleaned = re.sub(
        r"==================================.*?==================================",
        "",
        cleaned,
        flags=re.DOTALL,
    )
    return cleaned


def format_tool_messages(messages: str):
    if "<agent_response>" in messages:
        # cut only between <agent_response> and </agent_response>
        start = messages.find("<agent_response>") + len("<agent_response>")
        end = messages.find("</agent_response>")
        return messages[start:end].strip()
    elif "transfer_to_" in messages:
        if "math" in messages:
            return "Doing some math calculations..."
        elif "research" in messages:
            return "Searching information over internet..."
        elif "web_page" in messages:
            return "Reading web page..."
        elif "timing" in messages:
            return "Getting current time now..."
        elif "coding" in messages:
            return "Writing and compiling code..."
    return messages.strip()
