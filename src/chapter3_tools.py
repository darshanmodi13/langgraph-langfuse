from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from common import llm


# step1: tools
@tool
def search_document(query: str) -> str:
    """
    Search internal customer support knowledge base for documentation.
    Args: query: The search keywords or topics to lookup.
    """
    if "password" in query.lower():
        return "Docs Answer: Users can reset password via Settings -> Security -> Reset Password."
    elif "billing" in query.lower() or "charged" in query.lower():
        return "Docs Answer: Double charges are auto-refunded within 3-5 business days."
    return "Docs Answer: General user manual available at help.example.com."


@tool
def create_bug_ticket(description: str) -> str:
    """
    Create a bug ticket in internal issue tracker for technical crashes.
    Args: description: Summary description of technical issue.
    """
    return (
        f"Issue Tracker: Successfully logged bug ticket #BUG-8819 for '{description}'."
    )


tools = [search_document, create_bug_ticket]
llm_with_tools = llm.bind_tools(tools)


# step2: nodes
def llm_call(state: MessagesState) -> dict:
    """
    Call the LLM with the current state and return the response.
    """
    system_prompt = SystemMessage(
        content="You are a helpful customer support agent. Use tools to look up documentation or log bugs when needed."
    )
    response = llm_with_tools.invoke([system_prompt] + state.get("messages", []))
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Inspects last message to check if LLM emitted tool calls."""
    last_message = state.get("messages", [])[-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


# step3: graph
graph = StateGraph(MessagesState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
graph.add_edge("tool_node", "llm_call")

app = graph.compile()


def main() -> None:
    # step4: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # step5: execute graph
    query = {"messages": [HumanMessage(content="How do I reset my account password?")]}
    output = app.invoke(query, config={"callbacks": [CallbackHandler()]})

    print("--- AGENT DIALOGUE HISTORY ---")
    for m in output.get("messages", []):
        m.pretty_print()

    get_client().flush()


if __name__ == "__main__":
    main()
