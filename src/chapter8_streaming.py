from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from common import llm

POLICIES = {
    "billing": "Refunds are granted within 30 days of purchase.",
    "password": "Password resets require an email verification link.",
}


# step1: tools
@tool
def lookup_policy(topic: str) -> str:
    """Look up the company policy for a topic, such as billing or password."""
    # Sends a progress message to whoever is streaming with stream_mode="custom"
    get_stream_writer()({"status": f"Searching the policy database for '{topic}'..."})
    return POLICIES.get(topic.lower(), "Standard support procedures apply.")


tools = [lookup_policy]
llm_with_tools = llm.bind_tools(tools)


# step2: nodes
def agent(state: MessagesState) -> dict:
    get_stream_writer()({"status": "Asking the support assistant..."})
    system_prompt = SystemMessage(
        content="You are a helpful customer support agent. Use lookup_policy for policy questions."
    )
    response = llm_with_tools.invoke([system_prompt] + state.get("messages", []))
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    if getattr(state.get("messages", [])[-1], "tool_calls", None):
        return "tool_node"
    return END


# step3: graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "agent")

app = builder.compile()


def make_config(session_id: str) -> dict:
    return {
        "callbacks": [CallbackHandler()],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_trace_name": "streaming",
        },
    }


def main() -> None:
    # step4: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    question = {
        "messages": [
            HumanMessage(content="What is your refund policy for billing issues?")
        ]
    }

    # step5: stream_mode="updates": one chunk each time a node finishes
    print("===== updates: which node finished =====")
    for chunk in app.stream(
        question,
        config=make_config("stream-updates"),
        stream_mode="updates",
        version="v2",
    ):
        for node_name in chunk["data"]:
            print(f"[{node_name} finished]")

    # step6: stream_mode=["messages", "custom"]: LLM tokens as they arrive + our status messages
    print("\n===== messages + custom: live tokens and status =====")
    for chunk in app.stream(
        question,
        config=make_config("stream-tokens"),
        stream_mode=["messages", "custom"],
        version="v2",
    ):
        if chunk["type"] == "custom":
            print(f"\n[STATUS] {chunk['data'].get('status')}")
        elif chunk["type"] == "messages":
            token, metadata = chunk["data"]
            # Only the agent's text; skips tool results and empty tool-call chunks
            if metadata.get("langgraph_node") == "agent" and token.content:
                print(token.content, end="", flush=True)
    print("\n\n===== done =====")

    get_client().flush()


if __name__ == "__main__":
    main()
