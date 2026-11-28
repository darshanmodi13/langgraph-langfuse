import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from common import llm


# step1: state
class ReplyState(MessagesState):
    draft: str


# step2: nodes
async def draft_reply(state: ReplyState) -> dict:
    # async def + await: while waiting for Ollama, Python can run other work
    get_stream_writer()({"status": "Writing a draft reply..."})
    response = await llm.ainvoke(
        [
            SystemMessage(
                content="You are a customer support agent. Reply in 2-3 polite sentences."
            ),
            state.get("messages", [])[-1],
        ]
    )
    return {"draft": response.content}


def human_review(state: ReplyState) -> Command[Literal["send_reply", "cancel_reply"]]:
    # A plain def node is fine in an async graph: it awaits nothing
    decision = interrupt(
        {"question": "Send this reply to the customer?", "draft": state.get("draft")}
    )
    if decision == "yes":
        return Command(goto="send_reply")
    return Command(goto="cancel_reply")


def send_reply(state: ReplyState) -> dict:
    return {"messages": [AIMessage(content=state.get("draft", ""))]}


def cancel_reply(state: ReplyState) -> dict:
    return {"messages": [AIMessage(content="Reply cancelled by a human reviewer.")]}


# step3: graph
builder = StateGraph(ReplyState)
builder.add_node("draft_reply", draft_reply)
builder.add_node("human_review", human_review)
builder.add_node("send_reply", send_reply)
builder.add_node("cancel_reply", cancel_reply)

builder.add_edge(START, "draft_reply")
builder.add_edge("draft_reply", "human_review")
builder.add_edge("send_reply", END)
builder.add_edge("cancel_reply", END)

# step4: compile with a checkpointer (interrupts need one)
app = builder.compile(checkpointer=InMemorySaver())


# step5: stream one run; return the interrupt if the graph paused, else None
async def stream_run(graph_input: dict | Command, config: dict):
    paused = None
    async for chunk in app.astream(
        graph_input,
        config=config,
        stream_mode=["updates", "messages", "custom"],
        version="v2",
    ):
        if chunk["type"] == "custom":
            print(f"\n[STATUS] {chunk['data'].get('status')}")
        elif chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if metadata.get("langgraph_node") == "draft_reply" and token.content:
                print(token.content, end="", flush=True)
        elif chunk["type"] == "updates":
            # The pause arrives as an "updates" chunk with the key "__interrupt__"
            if "__interrupt__" in chunk["data"]:
                paused = chunk["data"]["__interrupt__"][0]
            else:
                for node_name in chunk["data"]:
                    print(f"\n[{node_name} finished]")
    return paused


# step6: stream, pause for the human, resume and stream again
async def handle_ticket(thread_id: str, text: str) -> None:
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [CallbackHandler()],
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_trace_name": "async-streaming",
        },
    }
    print(f"\n===== {thread_id} | customer: {text}")
    paused = await stream_run({"messages": [HumanMessage(content=text)]}, config)

    while paused:
        print(f"\n[GRAPH PAUSED] {paused.value.get('question')}")
        # input() blocks; to_thread runs it on another thread so the event loop stays free
        answer = await asyncio.to_thread(input, "Send? (yes/no): ")
        paused = await stream_run(Command(resume=answer.strip().lower()), config)

    state = await app.aget_state(config)
    print(f"\nfinal reply: {state.values.get('messages', [])[-1].content}")


async def main() -> None:
    # step7: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # step8: run one ticket
    await handle_ticket("ticket-801", "I was charged twice this month. Can you help?")

    get_client().flush()


if __name__ == "__main__":
    asyncio.run(main())
