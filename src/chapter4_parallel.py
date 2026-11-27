import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common import llm


# step1: schemas and structured llm
class DraftFeedback(BaseModel):
    grade: Literal["PASS", "FAIL"] = Field(
        description="Grade evaluation of draft response."
    )
    feedback: str = Field(description="Constructive feedback if grade is FAIL.")


llm_evaluator = llm.with_structured_output(DraftFeedback)


class SubTask(BaseModel):
    query_id: str
    question: str


# step2: state
class ParallelSupportState(TypedDict):
    ticket_text: str
    sub_tasks: list[SubTask]
    draft_answers: Annotated[list[str], operator.add]
    final_draft: str
    grade_status: str
    feedback: str
    attempts: int


MAX_ATTEMPTS = 3


class WorkerState(TypedDict):
    sub_task: SubTask


# step3: nodes
def orchestrator(state: ParallelSupportState) -> dict:
    """Breaks down complex ticket into sub-tasks."""
    tasks = [
        SubTask(query_id="1", question="How to update billing credit card?"),
        SubTask(query_id="2", question="Where to download VAT tax invoice?"),
    ]
    return {"sub_tasks": tasks}


def worker_answer(state: WorkerState) -> dict:
    """Worker node processing single sub-task in parallel."""
    task = state.get("sub_task")
    response = llm.invoke(
        [
            SystemMessage(content="Answer support query concisely."),
            HumanMessage(content=task.question),
        ]
    )
    return {"draft_answers": [f"[{task.query_id}] {response.content}"]}


def assign_workers(state: ParallelSupportState):
    """Dynamic fan-out routing function using Send API."""
    return [
        Send("worker_answer", {"sub_task": task}) for task in state.get("sub_tasks", [])
    ]


def synthesizer(state: ParallelSupportState) -> dict:
    """Synthesizes worker answers into single combined draft."""
    combined = "\n\n".join(state.get("draft_answers", []))
    return {"final_draft": combined}


def evaluate_draft(state: ParallelSupportState) -> dict:
    """Evaluator node evaluating draft response quality."""
    draft = state.get("final_draft", "")
    attempts = state.get("attempts", 0) + 1
    eval_result = llm_evaluator.invoke(
        [
            SystemMessage(
                content="Evaluate draft customer support response. Grade PASS if clear, FAIL if incomplete."
            ),
            HumanMessage(content=draft),
        ]
    )
    if eval_result is None:
        return {
            "grade_status": "PASS",
            "feedback": "Auto-passed fallback",
            "attempts": attempts,
        }
    return {
        "grade_status": eval_result.grade,
        "feedback": eval_result.feedback,
        "attempts": attempts,
    }


def route_evaluation(state: ParallelSupportState) -> Literal[END, "synthesizer"]:
    """Evaluator-optimizer conditional routing function."""
    if state.get("grade_status") == "PASS":
        return END
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return END
    return "synthesizer"


# step4: graph
builder = StateGraph(ParallelSupportState)
builder.add_node("orchestrator", orchestrator)
builder.add_node("worker_answer", worker_answer)
builder.add_node("synthesizer", synthesizer)
builder.add_node("evaluate_draft", evaluate_draft)
builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", assign_workers, ["worker_answer"])
builder.add_edge("worker_answer", "synthesizer")
builder.add_edge("synthesizer", "evaluate_draft")
builder.add_conditional_edges(
    "evaluate_draft", route_evaluation, {"synthesizer": "synthesizer", END: END}
)

app = builder.compile()


def main() -> None:
    # step5: graph visualization
    with open("graph.mmd", "w") as f:
        f.write(app.get_graph().draw_mermaid())

    # step6: execute graph
    result = app.invoke(
        {
            "ticket_text": "I need help updating my credit card and getting my monthly invoice."
        },
        config={"callbacks": [CallbackHandler()]},
    )

    print("--- FINAL EVALUATED DRAFT ---")
    print(result.get("final_draft"))
    print(f"Grade Status: {result.get('grade_status')}")
    print(f"Attempts: {result.get('attempts')}")

    get_client().flush()


if __name__ == "__main__":
    main()
