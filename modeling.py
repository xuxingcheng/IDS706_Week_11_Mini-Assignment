"""Train and evaluate the math pass/fail baseline."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline import read_students

ROOT = Path(__file__).resolve().parent
SEED = 42
FEATURES = ["math_g1", "math_g2", "portuguese_g1", "portuguese_g2"]
TARGET = "passed_math"


@dataclass(frozen=True)
class StudentSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_and_split() -> StudentSplit:
    """Keep each student in one stratified split, using an approximate 60/20/20 ratio."""
    students = read_students().sort_values("student_id").reset_index(drop=True)
    train, held_out = train_test_split(
        students, test_size=0.4, stratify=students[TARGET], random_state=SEED
    )
    validation, test = train_test_split(
        held_out, test_size=0.5, stratify=held_out[TARGET], random_state=SEED
    )
    return StudentSplit(train, validation, test)


def train_model(split: StudentSplit, model_name: str = "logistic_regression"):
    """Fit the requested classifier using training rows and earlier grades."""
    if model_name == "logistic_regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED))
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.fit(split.train[FEATURES], split.train[TARGET])


def _evaluate(model, students: pd.DataFrame) -> dict:
    expected = students[TARGET]
    predicted = model.predict(students[FEATURES])
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "f1": float(f1_score(expected, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=[0, 1]).tolist(),
    }


def evaluate_validation(model, split: StudentSplit) -> dict:
    """Score a trained candidate on the validation set."""
    return _evaluate(model, split.validation)


def evaluate_test(model, split: StudentSplit) -> dict:
    """Score the chosen model on the held-out test set."""
    return _evaluate(model, split.test)


def save_metrics(report: dict, path: Path) -> None:
    """Publish a complete JSON file so dashboard readers never see a partial write."""
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    Path(file.name).replace(path)


def save_confusion_matrix(metrics: dict, output_path: Path, title: str) -> None:
    """Save a labeled plot of a classifier's confusion matrix."""
    import numpy as np
    from matplotlib.figure import Figure

    figure = Figure(figsize=(5.2, 4.3), layout="constrained")
    axes = figure.subplots()
    ConfusionMatrixDisplay(
        np.asarray(metrics["confusion_matrix"]), display_labels=["Fail", "Pass"]
    ).plot(ax=axes, cmap="Blues", colorbar=False)
    axes.set_title(title)
    figure.savefig(output_path, dpi=150)


def run_baseline() -> dict:
    """Evaluate the fixed logistic baseline and save test results."""
    split = load_and_split()
    model = train_model(split)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(split.train[FEATURES], split.train[TARGET])
    report = {
        "model": "logistic_regression",
        "seed": SEED,
        "features": FEATURES,
        "target": TARGET,
        "positive_class": 1,
        "confusion_matrix_labels": [0, 1],
        "split_sizes": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "validation": evaluate_validation(model, split),
        "test": evaluate_test(model, split),
        "dummy_test": evaluate_test(dummy, split),
    }

    output_dir = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(report, output_dir / "baseline_metrics.json")
    save_confusion_matrix(
        report["test"],
        output_dir / "baseline_confusion_matrix.png",
        "Logistic regression on the test set",
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_baseline()
