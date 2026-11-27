import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from common import llm

# step1: setup env (.env and llm come from common.py)
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "development"


# step2: tools
@tool
def calculate_refund(amount: int, fee: int) -> int:
    """Calculates net refund amount given gross amount and processing fee."""
    return amount - fee


tools = [calculate_refund]
llm_with_tools = llm.bind_tools(tools)


# step3: nodes
def llm_call(state: MessagesState):
    system_prompt = SystemMessage(content="You are an automated support assistant.")
    response = llm_with_tools.invoke([system_prompt] + state.get("messages", []))
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal[END, "tool_node"]:
    if getattr(state.get("messages", [])[-1], "tool_calls", None):
        return "tool_node"
    return END


# step4: graph
builder = StateGraph(MessagesState)

builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
app = builder.compile()


def main() -> None:
    # step5: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # step6: execute graph
    input_payload = {
        "messages": [
            HumanMessage(content="Calculate refund for amount 100 with fee 15.")
        ]
    }
    config = {
        "callbacks": [CallbackHandler()],
        "metadata": {
            "langfuse_session_id": "ticket-123",
            "langfuse_user_id": "customer-42",
            "langfuse_tags": ["billing", "refund"],
            "langfuse_trace_name": "support-ticket",
        },
    }

    response = app.invoke(input_payload, config=config)
    print("Final Reply:", response.get("messages", [])[-1].content)

    get_client().flush()


if __name__ == "__main__":
    main()
