import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from common import llm


# step1: state
class BillingState(MessagesState):
    ticket_type: str
    amount: float
    draft_response: str
    approval_status: str


# step2: nodes
def classify_ticket(state: BillingState) -> dict:
    text = state.get("messages", [])[-1].content
    if "refund" in text.lower() or "billing" in text.lower():
        match = re.search(r"\$(\d+(?:\.\d+)?)", text)
        return {
            "ticket_type": "billing",
            "amount": float(match.group(1)) if match else 0.0,
        }
    return {"ticket_type": "general", "amount": 0.0}


def draft_response(state: BillingState) -> dict:
    # The LLM call has its own node: on resume only human_review runs again, so this is not repeated
    ticket = state.get("messages", [])[-1].content
    if state.get("ticket_type") == "billing":
        task = f"Write a reply confirming a refund of ${state.get('amount', 0.0):.2f}."
    else:
        task = "Write a reply to this support message."
    response = llm.invoke(
        [
            SystemMessage(
                content="You are a customer support agent. Reply in 2-3 polite sentences."
            ),
            HumanMessage(content=f"{task}\n\nCustomer message: {ticket}"),
        ]
    )
    return {"draft_response": response.content}


def human_review(state: BillingState) -> Command[Literal["send_reply", "cancel_reply"]]:
    if state.get("ticket_type") != "billing":
        return Command(update={"approval_status": "auto_approved"}, goto="send_reply")

    # 1st run: pauses here. After Command(resume=...): returns the reviewer's answer
    decision = interrupt(
        {
            "question": "Approve this refund reply?",
            "amount": state.get("amount"),
            "draft": state.get("draft_response"),
        }
    )
    if decision.get("approved"):
        final_draft = decision.get("edited_draft", state.get("draft_response", ""))
        return Command(
            update={"draft_response": final_draft, "approval_status": "approved"},
            goto="send_reply",
        )
    return Command(update={"approval_status": "rejected"}, goto="cancel_reply")


def send_reply(state: BillingState) -> dict:
    return {"messages": [AIMessage(content=state.get("draft_response", ""))]}


def cancel_reply(state: BillingState) -> dict:
    return {"messages": [AIMessage(content="Action cancelled by human reviewer.")]}


# step3: graph
builder = StateGraph(BillingState)
builder.add_node("classify_ticket", classify_ticket)
builder.add_node("draft_response", draft_response)
builder.add_node("human_review", human_review)
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
    print("\n[GRAPH PAUSED: waiting for a human]")
    print(f"  {question.get('question')}  amount: ${question.get('amount', 0.0):.2f}")
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

    # step6: a billing ticket pauses for approval
    handle_ticket(
        "ticket-501", "customer-42", "I was double charged. Please refund $49.99."
    )

    # step7: a general ticket is auto-approved, no pause
    handle_ticket("ticket-502", "customer-77", "How do I change my profile picture?")

    get_client().flush()


if __name__ == "__main__":
    main()
