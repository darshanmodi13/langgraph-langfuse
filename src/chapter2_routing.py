from typing import Literal, TypedDict

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import common  # noqa: F401  loads .env; this step uses keyword rules, not the llm


# step1: state
class TicketClassification(TypedDict):
    intent: Literal["billing", "bug", "question"]
    urgency: Literal["low", "medium", "high"]
    summary: str


class SupportTicketState(TypedDict):
    email_content: str
    classification: TicketClassification | None
    search_results: list[str] | None
    draft_response: str | None


# step2: nodes
def classify_intent(
    state: SupportTicketState,
) -> Command[Literal["search_documentation", "bug_tracking", "draft_response"]]:
    """Classifies ticket intent and returns a Command combining state update and routing target."""
    content = state.get("email_content", "").lower()

    # Deterministic classification logic based on content inspection
    if "charged" in content or "billing" in content or "refund" in content:
        classification: TicketClassification = {
            "intent": "billing",
            "urgency": "high",
            "summary": "Customer reports subscription billing issue.",
        }
        goto = "draft_response"
    elif "crash" in content or "bug" in content or "error" in content:
        classification = {
            "intent": "bug",
            "urgency": "medium",
            "summary": "Customer reports technical software bug.",
        }
        goto = "bug_tracking"
    else:
        classification = {
            "intent": "question",
            "urgency": "low",
            "summary": "General product documentation question.",
        }
        goto = "search_documentation"

    # Return Command object containing partial state update AND explicit next destination
    return Command(update={"classification": classification}, goto=goto)


def search_documentation(
    state: SupportTicketState,
) -> Command[Literal["draft_response"]]:
    """Retrieves documentation context and routes directly to drafting."""
    return Command(
        update={
            "search_results": [
                "Docs: Password reset instructions available at help/reset"
            ]
        },
        goto="draft_response",
    )


def bug_tracking(state: SupportTicketState) -> Command[Literal["draft_response"]]:
    """Creates a tracking ticket and routes directly to drafting."""
    return Command(
        update={"search_results": ["Bug tracker: Created issue ticket #BUG-9921"]},
        goto="draft_response",
    )


def draft_response(state: SupportTicketState) -> dict:
    """Drafts final customer support response based on accumulated state context."""
    classification = state.get("classification") or {}
    results = state.get("search_results") or []
    draft = f"Hello! Thanks for reaching out about your {classification.get('intent', 'request')} issue."
    if results:
        draft += f" Context update: {results[0]}"
    return {"draft_response": draft}


# step3: graph
builder = StateGraph(SupportTicketState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("search_documentation", search_documentation)
builder.add_node("bug_tracking", bug_tracking)
builder.add_node("draft_response", draft_response)

builder.add_edge(START, "classify_intent")
builder.add_edge("draft_response", END)
app = builder.compile()

# step4: graph visualization
with open("graph.mmd", "w") as f:
    f.write(app.get_graph().draw_mermaid())

# step5: execute graph
input_payload = {
    "email_content": "I was charged twice for my subscription this month!",
    "classification": None,
    "search_results": None,
    "draft_response": None,
}
output = app.invoke(input_payload, config={"callbacks": [CallbackHandler()]})

classification = output.get("classification") or {}
print("--- GRAPH EXECUTION RESULT ---")
print(f"Intent Classified : {classification.get('intent')}")
print(f"Urgency Level : {classification.get('urgency')}")
print(f"Final Response : {output.get('draft_response')}")

get_client().flush()
