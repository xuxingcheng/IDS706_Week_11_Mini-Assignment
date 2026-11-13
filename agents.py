"""Adapt the class LangGraph research loop to student pass/fail prediction."""

import argparse
import json
import operator
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sklearn.base import BaseEstimator

from modeling import (
    FEATURES,
    ROOT,
    SEED,
    TARGET,
    StudentSplit,
    evaluate_test,
    evaluate_validation,
    load_and_split,
    save_confusion_matrix,
    save_metrics,
    train_model,
)

DEFAULT_THRESHOLD = 0.97
CANDIDATES = (
    ("logistic_regression", "Start with a linear model using the four earlier grades."),
    ("decision_tree", "Try a shallow tree to capture grade cutoffs and interactions."),
)


class WorkflowState(TypedDict, total=False):
    threshold: float
    split: StudentSplit
    split_sizes: dict[str, int]
    candidate: str
    models: dict[str, BaseEstimator]
    history: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    retry: bool
    stop_reason: str
    result: dict


def output_directory() -> Path:
    return Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))


def data_agent(state: WorkflowState) -> dict:
    split = load_and_split()
    pass_rate = split.train[TARGET].mean()
    sizes = {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }
    return {
        "split": split,
        "split_sizes": sizes,
        "models": {},
        "trace": [{
            "node": "data_agent",
            "message": (
                f"Loaded {sum(sizes.values())} students from SQLite. "
                f"Split into {sizes['train']} train, {sizes['validation']} validation, "
                f"and {sizes['test']} test rows. "
                f"Training pass rate was {pass_rate:.1%}. Model inputs are the four earlier grades."
            ),
        }],
    }


def scientist(state: WorkflowState) -> dict:
    model_name, hypothesis = CANDIDATES[len(state["history"])]
    if state["history"]:
        last_f1 = state["history"][-1]["validation"]["f1"]
        hypothesis = f"Previous validation F1 was {last_f1:.3f}. {hypothesis}"
    return {
        "candidate": model_name,
        "trace": [{"node": "scientist", "message": hypothesis}],
    }


def engineer(state: WorkflowState) -> dict:
    name = state["candidate"]
    model = train_model(state["split"], name)
    return {
        "models": {**state["models"], name: model},
        "trace": [{"node": "engineer", "message": f"Fitted {name} on the training split."}],
    }


def critic(state: WorkflowState) -> dict:
    name = state["candidate"]
    validation = evaluate_validation(state["models"][name], state["split"])
    target_met = validation["f1"] >= state["threshold"]
    retry = not target_met and len(state["history"]) + 1 < len(CANDIDATES)
    if target_met:
        reason = "Validation F1 reached the target."
    elif retry:
        reason = "Validation F1 is below the target. Try the remaining candidate."
    else:
        reason = "Both candidates ran and validation F1 stayed below the target."
    return {
        "history": [{"model": name, "validation": validation, "decision": reason}],
        "retry": retry,
        "stop_reason": reason,
        "trace": [{
            "node": "critic",
            "message": f"{name} validation F1 was {validation['f1']:.3f}. {reason}",
        }],
    }


def route_after_critic(state: WorkflowState) -> str:
    return "scientist" if state["retry"] else "report"


def report(state: WorkflowState) -> dict:
    # max() keeps the first candidate on an F1 tie.
    best = max(state["history"], key=lambda trial: trial["validation"]["f1"])
    name = best["model"]
    test = evaluate_test(state["models"][name], state["split"])
    event = {
        "node": "report",
        "message": f"Selected {name} by validation F1 and evaluated it once on the test split.",
    }
    result = {
        "model": name,
        "seed": SEED,
        "features": FEATURES,
        "target": TARGET,
        "positive_class": 1,
        "confusion_matrix_labels": [0, 1],
        "split_sizes": state["split_sizes"],
        "threshold": state["threshold"],
        "stop_reason": state["stop_reason"],
        "validation": best["validation"],
        "test": test,
        "history": state["history"],
        "trace": [*state["trace"], event],
    }
    output_dir = output_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(test, output_dir / "agent_confusion_matrix.png", "Selected model on the test set")
    save_metrics(result, output_dir / "agent_metrics.json")
    return {"result": result, "trace": [event]}


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("data_agent", data_agent)
    graph.add_node("scientist", scientist)
    graph.add_node("engineer", engineer)
    graph.add_node("critic", critic)
    graph.add_node("report", report)
    graph.add_edge(START, "data_agent")
    graph.add_edge("data_agent", "scientist")
    graph.add_edge("scientist", "engineer")
    graph.add_edge("engineer", "critic")
    graph.add_conditional_edges("critic", route_after_critic, ["scientist", "report"])
    graph.add_edge("report", END)
    return graph.compile()


def stream_workflow(
    threshold: float = DEFAULT_THRESHOLD, graph: CompiledStateGraph | None = None
) -> Iterator[dict]:
    """Yield completed LangGraph node events and include results in the report event."""
    if not 0 <= threshold <= 1:
        raise ValueError("Validation F1 target must be between 0 and 1.")
    if graph is None:
        graph = build_graph()
    for update in graph.stream({"threshold": threshold}, stream_mode="updates"):
        for patch in update.values():
            event = patch["trace"][0]
            yield {**event, "result": patch["result"]} if "result" in patch else event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Validation F1 target")
    for step in stream_workflow(parser.parse_args().threshold):
        print(step["message"], flush=True)
        if step["node"] == "report":
            print(json.dumps(step["result"]["test"], indent=2))
