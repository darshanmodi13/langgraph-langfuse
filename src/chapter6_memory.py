from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import tool
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from common import llm

MAX_HISTORY_TOKENS = 1000


# step1: context (who is chatting; passed with every invoke, not saved in state)
@dataclass
class Context:
    user_id: str


# step2: tools
@tool
def reset_password(username: str) -> str:
    """Resets the account password for a given username."""
    return f"Password reset instructions sent for user {username}."


tools = [reset_password]
llm_with_tools = llm.bind_tools(tools)


# step3: nodes
def llm_call(state: MessagesState, runtime: Runtime[Context]) -> dict:
    # Long-term memory: facts saved for this user, shared by all their threads
    saved = runtime.store.search(("users", runtime.context.user_id))
    profile = (
        "\n".join(f"- {item.key}: {item.value}" for item in saved)
        or "- nothing saved yet"
    )

    system_prompt = SystemMessage(
        content=f"You are a helpful customer support assistant.\nWhat we know about this customer:\n{profile}"
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


# step4: graph
builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")


# step5: memory
checkpointer = InMemorySaver()  # short-term: conversation state per thread_id
store = InMemoryStore()  # long-term: facts per user, shared across threads

# step6: compile
app = builder.compile(checkpointer=checkpointer, store=store)


def chat(thread_id: str, user_id: str, text: str) -> str:
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [CallbackHandler()],
        "metadata": {
            # Langfuse does not group by thread_id on its own
            "langfuse_session_id": thread_id,
            "langfuse_user_id": user_id,
            "langfuse_trace_name": "support-chat",
        },
    }
    result = app.invoke(
        {"messages": [HumanMessage(content=text)]},
        config=config,
        context=Context(user_id=user_id),
    )
    return result.get("messages", [])[-1].content


def main() -> None:
    # step7: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # step8: a 3-turn conversation on thread T-1 (remembered by the checkpointer)
    for text in [
        "Hi, my username is alex_dev.",
        "Please reset my password.",
        "What is my username?",
    ]:
        print(f"\n[T-1 | customer-42] user: {text}")
        print(f"[T-1 | customer-42] bot : {chat('T-1', 'customer-42', text)}")

    # step9: save long-term facts for customer-42 (in a real app a node or tool would do this)
    store.put(
        ("users", "customer-42"), "profile", {"username": "alex_dev", "plan": "pro"}
    )

    # step10: new thread, same customer: no chat history, but the store knows them
    question = "What is my username and which plan am I on?"
    print(f"\n[T-2 | customer-42] user: {question}")
    print(f"[T-2 | customer-42] bot : {chat('T-2', 'customer-42', question)}")

    # step11: new thread, different customer: nothing in either memory
    print(f"\n[T-3 | customer-99] user: {question}")
    print(f"[T-3 | customer-99] bot : {chat('T-3', 'customer-99', question)}")

    # step12: what the checkpointer (InMemorySaver) holds
    print("\n===== InMemorySaver (checkpointer) =====")
    for thread_id in ["T-1", "T-2", "T-3"]:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = list(checkpointer.list(config))
        messages = app.get_state(config).values.get("messages", [])
        print(
            f"{thread_id}: {len(checkpoints)} checkpoints, latest state has {len(messages)} messages"
        )
    print("Latest state of T-1:")
    for message in app.get_state({"configurable": {"thread_id": "T-1"}}).values.get(
        "messages", []
    ):
        print(f"  {message.type:<6} {str(message.content)[:60]}")

    # step13: what the store (InMemoryStore) holds
    print("\n===== InMemoryStore (store) =====")
    for item in store.search(("users",)):
        print(f"  namespace={item.namespace} key={item.key!r} value={item.value!r}")

    get_client().flush()


if __name__ == "__main__":
    main()
