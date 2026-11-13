"""Local Streamlit dashboard for the student pass/fail research loop."""

import json
import sqlite3

import streamlit as st

from agents import DEFAULT_THRESHOLD, build_graph, output_directory, stream_workflow

ROLE_LABELS = {
    "__start__": "Start",
    "data_agent": "Data agent",
    "scientist": "Scientist",
    "engineer": "Engineer",
    "critic": "Critic",
    "report": "Report",
    "__end__": "End",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Shallow decision tree",
}


def graph_dot(graph, completed: set[str], latest: str | None = None) -> str:
    dot = [
        "digraph {",
        "rankdir=LR",
        'node [shape=box style="rounded,filled" color="#94a3b8" fontcolor="#172554"]',
        'edge [color="#64748b" fontcolor="#475569"]',
    ]
    for node in graph.nodes:
        color = "#93c5fd" if node == latest else "#dcfce7" if node in completed else "#f8fafc"
        dot.append(f'"{node}" [label="{ROLE_LABELS[node]}" fillcolor="{color}"]')
    for edge in graph.edges:
        label = ""
        if edge.conditional:
            label = ' [label="Try another model"]' if edge.target == "scientist" else ' [label="Finish"]'
        dot.append(f'"{edge.source}" -> "{edge.target}"{label}')
    return "\n".join([*dot, "}"])


def show_results(result: dict) -> None:
    st.subheader("Results")
    model = MODEL_LABELS[result["model"]]
    st.write(f"**{model}** selected by validation F1. {result['stop_reason']}")
    st.caption(f"Validation F1 target for this run {result['threshold']:.2f}")

    st.dataframe(
        [
            {
                "Candidate": MODEL_LABELS[trial["model"]],
                "Validation accuracy": trial["validation"]["accuracy"],
                "Validation F1": trial["validation"]["f1"],
            }
            for trial in result["history"]
        ],
        hide_index=True,
        column_config={
            "Validation accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Validation F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    comparison = [{"Model": f"Selected · {model}", **result["test"]}]
    baseline_path = output_directory() / "baseline_metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison.append({"Model": "Part 1 · Logistic regression", **baseline["test"]})
        comparison.append({"Model": "Part 1 · Majority class", **baseline["dummy_test"]})
    else:
        st.caption("Run the Airflow DAG to generate Part 1 baseline metrics.")
    st.dataframe(
        [
            {"Model": row["Model"], "Test accuracy": row["accuracy"], "Test F1": row["f1"]}
            for row in comparison
        ],
        hide_index=True,
        column_config={
            "Test accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Test F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    matrix_path = output_directory() / "agent_confusion_matrix.png"
    if matrix_path.exists():
        st.image(str(matrix_path), width=440)


def main() -> None:
    st.set_page_config(page_title="Student performance lab", layout="wide")
    st.title("Student performance lab")
    st.write("Predict whether a student passes math using earlier grades from both courses.")
    threshold = st.number_input(
        "Validation F1 target", min_value=0.0, max_value=1.0,
        value=DEFAULT_THRESHOLD, step=0.01, format="%.2f",
    )
    run = st.button("Run agent workflow", type="primary")
    graph = build_graph()
    diagram = graph.get_graph()
    canvas = st.empty()
    canvas.graphviz_chart(graph_dot(diagram, set()), width="stretch")
    result = None

    if run:
        completed = {"__start__"}
        with st.status("Running the workflow", expanded=True) as status:
            try:
                for event in stream_workflow(threshold, graph):
                    node = event["node"]
                    completed.add(node)
                    canvas.graphviz_chart(graph_dot(diagram, completed, node), width="stretch")
                    st.write(f"**{ROLE_LABELS[node]}** · {event['message']}")
                    if node == "report":
                        result = event["result"]
            except (sqlite3.Error, ValueError, OSError) as exc:
                status.update(label="Workflow failed", state="error")
                st.error(f"{exc}. Check that the Airflow DAG completed and run again.")
            else:
                completed.add("__end__")
                canvas.graphviz_chart(graph_dot(diagram, completed, "report"), width="stretch")
                status.update(label="Workflow completed", state="complete")
    else:
        result_path = output_directory() / "agent_metrics.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            with st.expander("Last workflow trace"):
                for event in result["trace"]:
                    st.write(f"**{ROLE_LABELS[event['node']]}** · {event['message']}")

    if result is not None:
        show_results(result)


if __name__ == "__main__":
    main()
