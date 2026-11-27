import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from common import llm

APPROVAL_LIMIT = 50.0  # refunds above this need a manager
FINANCE_LIMIT = 500.0  # refunds above this also need finance


# step1: state
class BillingState(MessagesState):
    ticket_type: str
    amount: float | None  # None = no amount found in the message
    draft_response: str
    approval_status: str


# step2: nodes
def classify_ticket(state: BillingState) -> dict:
    text = state.get("messages", [])[-1].content
    if "refund" in text.lower() or "billing" in text.lower():
        match = re.search(r"\$(\d+(?:\.\d+)?)", text)
        return {
            "ticket_type": "billing",
            "amount": float(match.group(1)) if match else None,
        }
    return {"ticket_type": "general", "amount": None}


def draft_response(state: BillingState) -> dict:
    # The LLM call has its own node: on resume only the review nodes run again, so this is not repeated
    ticket = state.get("messages", [])[-1].content
    amount = state.get("amount")
    if state.get("ticket_type") != "billing":
        task = "Write a reply to this support message."
    elif amount is None:
        task = "Write a reply saying we are reviewing the refund request."
    else:
        task = f"Write a reply confirming a refund of ${amount:.2f}."
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a customer support agent. Reply in 2-3 polite sentences."
            ),
            HumanMessage(content=f"{task}\n\nCustomer message: {ticket}"),
        ]
    )
    return {"draft_response": response.content}


def human_review(
    state: BillingState,
) -> Command[Literal["finance_review", "send_reply", "cancel_reply"]]:
    amount = state.get("amount")
    if state.get("ticket_type") != "billing":
        return Command(update={"approval_status": "auto_approved"}, goto="send_reply")
    # Safe before interrupt(): it only reads state, which is the same when the node re-runs on resume
    if amount is not None and amount <= APPROVAL_LIMIT:
        return Command(update={"approval_status": "auto_approved"}, goto="send_reply")

    decision = interrupt(
        {
            "reviewer": "manager",
            "question": "Approve this refund reply?",
            "amount": amount,
            "draft": state.get("draft_response"),
        }
    )
    if not decision.get("approved"):
        return Command(update={"approval_status": "rejected"}, goto="cancel_reply")

    final_draft = decision.get("edited_draft", state.get("draft_response", ""))
    if amount is not None and amount > FINANCE_LIMIT:
        return Command(
            update={
                "draft_response": final_draft,
                "approval_status": "manager_approved",
            },
            goto="finance_review",
        )
    return Command(
        update={"draft_response": final_draft, "approval_status": "approved"},
        goto="send_reply",
    )


def finance_review(
    state: BillingState,
) -> Command[Literal["send_reply", "cancel_reply"]]:
    # A separate node, so the manager's answer is not replayed: each approval is its own pause
    decision = interrupt(
        {
            "reviewer": "finance",
            "question": "Manager approved. Approve this large refund?",
            "amount": state.get("amount"),
            "draft": state.get("draft_response"),
        }
    )
    if not decision.get("approved"):
        return Command(update={"approval_status": "rejected"}, goto="cancel_reply")
    final_draft = decision.get("edited_draft", state.get("draft_response", ""))
    return Command(
        update={"draft_response": final_draft, "approval_status": "approved"},
        goto="send_reply",
    )


def send_reply(state: BillingState) -> dict:
    return {"messages": [AIMessage(content=state.get("draft_response", ""))]}


def cancel_reply(state: BillingState) -> dict:
    return {"messages": [AIMessage(content="Action cancelled by human reviewer.")]}


# step3: graph
builder = StateGraph(BillingState)
builder.add_node("classify_ticket", classify_ticket)
builder.add_node("draft_response", draft_response)
builder.add_node("human_review", human_review)
builder.add_node("finance_review", finance_review)
builder.add_node("send_reply", send_reply)
builder.add_node("cancel_reply", cancel_reply)

builder.add_edge(START, "classify_ticket")
builder.add_edge("classify_ticket", "draft_response")
builder.add_edge("draft_response", "human_review")
builder.add_edge("send_reply", END)
builder.add_edge("cancel_reply", END)

# step4: compile with a checkpointer (interrupts need one to save the paused state)
app = builder.compile(checkpointer=InMemorySaver())


def ask_reviewer(question: dict) -> dict:
    amount = question.get("amount")
    shown_amount = "unknown" if amount is None else f"${amount:.2f}"
    print(f"\n[GRAPH PAUSED: waiting for {question.get('reviewer')}]")
    print(f"  {question.get('question')}  amount: {shown_amount}")
    print(f"  draft: {question.get('draft')}")
    choice = input("Approve? (yes/edit/no): ").strip().lower()
    if choice == "yes":
        return {"approved": True}
    if choice == "edit":
        return {"approved": True, "edited_draft": input("Enter the new reply: ")}
    return {"approved": False}


def handle_ticket(thread_id: str, user_id: str, text: str) -> None:
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [CallbackHandler()],
        "metadata": {
            # The first run and each resume are separate traces; one session groups them
            "langfuse_session_id": thread_id,
            "langfuse_user_id": user_id,
            "langfuse_trace_name": "billing-review",
        },
    }
    print(f"\n===== {thread_id} | customer: {text}")
    result = app.invoke({"messages": [HumanMessage(content=text)]}, config=config)

    while "__interrupt__" in result:
        answer = ask_reviewer(result["__interrupt__"][0].value)
        result = app.invoke(Command(resume=answer), config=config)

    print(f"status: {result.get('approval_status')}")
    print(f"reply : {result.get('messages', [])[-1].content}")


def main() -> None:
    # step5: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # step6: one ticket per rule
    tickets = [
        ("ticket-601", "customer-41", "Please refund $30."),  # auto-approved
        (
            "ticket-602",
            "customer-42",
            "Please refund $50.",
        ),  # auto-approved: limit is "more than $50"
        (
            "ticket-603",
            "customer-43",
            "I was double charged. Please refund $150.",
        ),  # manager
        (
            "ticket-604",
            "customer-44",
            "Please refund $600 for my annual plan.",
        ),  # manager, then finance
        (
            "ticket-605",
            "customer-45",
            "Please refund my last payment.",
        ),  # unknown amount: manager
        (
            "ticket-606",
            "customer-46",
            "How do I change my profile picture?",
        ),  # general: auto-approved
    ]
    for thread_id, user_id, text in tickets:
        handle_ticket(thread_id, user_id, text)

    get_client().flush()


if __name__ == "__main__":
    main()
