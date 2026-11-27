from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

llm = ChatOllama(model="qwen3.5:9b", temperature=0, reasoning=False)


class TicketClassification(TypedDict):
    intent: Literal["billing", "bug", "question"]
    urgency: Literal["low", "high"]


def classify_initial_ticket(state: MessagesState) -> dict:
    """Reads the user message from state and returns an initial AI classification."""
    user_message = state["messages"][-1].content
    prompt = f"Categorize this support ticket briefly (e.g., Billing, Bug, General): {user_message}"
    response = llm.invoke([HumanMessage(content=prompt)])
    # Return a partial dictionary update to append the new message to state
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("classify_init_ticket", classify_initial_ticket)
builder.add_edge(START, "classify_init_ticket")
builder.add_edge("classify_init_ticket", END)

app = builder.compile()

initial_state = {
    "messages": [
        HumanMessage(content="I was double charged for my monthly subscription!")
    ]
}
langfuse_handler = CallbackHandler()
result = app.invoke(initial_state, config={"callbacks": [langfuse_handler]})

for msg in result["messages"]:
    msg.pretty_print()

get_client().flush()
