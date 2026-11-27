from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, MessagesState, StateGraph

from common import llm


# step1: nodes
def classify_initial_ticket(state: MessagesState) -> dict:
    """Reads the user message from state and returns an initial AI classification."""
    user_message = state.get("messages", [])[-1].content
    prompt = f"Categorize this support ticket briefly (e.g., Billing, Bug, General): {user_message}"
    response = llm.invoke([HumanMessage(content=prompt)])
    # Return a partial dictionary update to append the new message to state
    return {"messages": [response]}


# step2: graph
builder = StateGraph(MessagesState)

builder.add_node("classify_init_ticket", classify_initial_ticket)
builder.add_edge(START, "classify_init_ticket")
builder.add_edge("classify_init_ticket", END)

app = builder.compile()


def main() -> None:
    # step3: execute graph
    initial_state = {
        "messages": [
            HumanMessage(content="I was double charged for my monthly subscription!")
        ]
    }
    result = app.invoke(initial_state, config={"callbacks": [CallbackHandler()]})

    for msg in result.get("messages", []):
        msg.pretty_print()

    get_client().flush()


if __name__ == "__main__":
    main()
