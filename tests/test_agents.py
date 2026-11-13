"""Check routing and holdout use in the Part 2 graph."""

import json
import sqlite3
from unittest.mock import Mock

import pandas as pd
import pytest

import agents
from modeling import TARGET, StudentSplit


@pytest.fixture
def simulated_run(monkeypatch, tmp_path):
    frame = pd.DataFrame({"student_id": [1], TARGET: [1]})
    split = StudentSplit(frame, frame, frame)
    model_scores = {"logistic_regression": 0.9, "decision_tree": 0.8}
    holdout = Mock(return_value={"accuracy": 0.5, "f1": 0.5, "confusion_matrix": [[1, 1], [1, 1]]})
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(agents, "load_and_split", lambda: split)
    monkeypatch.setattr(agents, "train_model", lambda split, name: name)
    monkeypatch.setattr(
        agents, "evaluate_validation", lambda name, split: {"f1": model_scores[name]}
    )
    monkeypatch.setattr(agents, "evaluate_test", holdout)
    monkeypatch.setattr(agents, "save_confusion_matrix", Mock())
    return model_scores, holdout, split


def test_critic_accepts_when_threshold_is_met(simulated_run):
    _, holdout, split = simulated_run
    events = list(agents.stream_workflow(threshold=0.9))

    assert [event["node"] for event in events] == [
        "data_agent", "scientist", "engineer", "critic", "report",
    ]
    assert events[-1]["result"]["model"] == "logistic_regression"
    assert "reached" in events[-1]["result"]["stop_reason"]
    holdout.assert_called_once_with("logistic_regression", split)


@pytest.mark.parametrize(
    ("tree_f1", "expected_model"),
    [(0.8, "logistic_regression"), (0.9, "logistic_regression"), (0.95, "decision_tree")],
)
def test_loop_stops_after_two_and_scores_only_best_candidate(
    simulated_run, tmp_path, tree_f1, expected_model
):
    scores, holdout, split = simulated_run
    scores["decision_tree"] = tree_f1
    events = list(agents.stream_workflow(threshold=1.0))
    result = events[-1]["result"]

    assert [event["node"] for event in events] == [
        "data_agent", "scientist", "engineer", "critic",
        "scientist", "engineer", "critic", "report",
    ]
    assert [trial["model"] for trial in result["history"]] == [
        "logistic_regression", "decision_tree",
    ]
    assert [trial["validation"]["f1"] for trial in result["history"]] == [0.9, tree_f1]
    assert result["model"] == expected_model
    assert "stayed below" in result["stop_reason"]
    assert result["trace"] == [{key: event[key] for key in ("node", "message")} for event in events]
    assert json.loads((tmp_path / "agent_metrics.json").read_text()) == result
    holdout.assert_called_once_with(expected_model, split)


def test_database_failure_reaches_the_caller(simulated_run, monkeypatch, tmp_path):
    _, holdout, _ = simulated_run
    monkeypatch.setattr(
        agents, "load_and_split", Mock(side_effect=sqlite3.OperationalError("unable to open database file"))
    )

    with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
        list(agents.stream_workflow())

    holdout.assert_not_called()
    assert not (tmp_path / "agent_metrics.json").exists()
