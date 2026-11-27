from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import tool
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.memory import InMemoryStore

from common import llm

MAX_HISTORY_TOKENS = 1000


# step1: tools
@tool
def reset_password(username: str) -> str:
    """Resets the account password for a given username."""
    return f"Password reset instructions sent for user {username}."


tools = [reset_password]
llm_with_tools = llm.bind_tools(tools)


# step2: nodes
def llm_call(state: MessagesState) -> dict:
    system_prompt = SystemMessage(
        content="You are a helpful customer support assistant. Remember details the customer tells you."
    )
    # The checkpointer keeps the full history; only the newest part is sent to the LLM
    history = trim_messages(
        state.get("messages", []),
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_HISTORY_TOKENS,
        start_on="human",
    )
    response = llm_with_tools.invoke([system_prompt] + history)
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    if getattr(state.get("messages", [])[-1], "tool_calls", None):
        return "tool_node"
    return END


# step3: graph with a checkpointer (short-term memory per thread_id)
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")


# step4: Memory
checkpointer = InMemorySaver()
store = InMemoryStore()

# step5:compile
app = builder.compile(checkpointer=checkpointer)


def chat(thread_id: str, text: str) -> str:
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [CallbackHandler()],
        "metadata": {
            # Langfuse does not group by thread_id on its own
            "langfuse_session_id": thread_id,
            "langfuse_trace_name": "support-chat",
        },
    }
    result = app.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return result.get("messages", [])[-1].content


def main() -> None:
    # step6: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # step7: a 3-turn conversation on thread T-1
    for text in [
        "Hi, my username is alex_dev.",
        "Please reset my password.",
        "What is my username?",
    ]:
        print(f"\n[T-1] user: {text}")
        print(f"[T-1] bot : {chat('T-1', text)}")

    # step8: the same question on a new thread T-2 (no shared memory)
    print("\n[T-2] user: What is my username?")
    print(f"[T-2] bot : {chat('T-2', 'What is my username?')}")

    # step9: inspect what the checkpointer saved for T-1
    snapshot = app.get_state({"configurable": {"thread_id": "T-1"}})
    print(f"\nSaved messages in T-1: {len(snapshot.values.get('messages', []))}")
    for message in snapshot.values.get("messages", []):
        print(f"  {message.type:<6} {str(message.content)[:60]}")

    get_client().flush()


if __name__ == "__main__":
    main()
