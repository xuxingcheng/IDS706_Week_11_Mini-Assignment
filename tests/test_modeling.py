import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import modeling


def student_rows():
    grades = [i % 20 for i in range(100)]
    return pd.DataFrame({
        "student_id": [f"student-{i:03}" for i in range(len(grades))],
        "math_g1": grades,
        "math_g2": grades,
        "portuguese_g1": grades,
        "portuguese_g2": grades,
        "math_final": grades,
        "passed_math": [int(grade >= 10) for grade in grades],
    })


class ModelingTests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_independent_of_source_order(self):
        students = student_rows()
        with patch.object(modeling, "read_students", return_value=students):
            split = modeling.load_and_split()
        with patch.object(modeling, "read_students", return_value=students.iloc[::-1]):
            reordered = modeling.load_and_split()

        student_ids = []
        for name, expected_size in (("train", 60), ("validation", 20), ("test", 20)):
            part = getattr(split, name)
            self.assertEqual(len(part), expected_size)
            self.assertEqual(part[modeling.TARGET].mean(), 0.5)
            self.assertEqual(
                part["student_id"].tolist(), getattr(reordered, name)["student_id"].tolist()
            )
            student_ids.extend(part["student_id"])
        self.assertCountEqual(student_ids, students["student_id"])

    def test_models_only_fit_earlier_grades(self):
        with patch.object(modeling, "read_students", return_value=student_rows()):
            split = modeling.load_and_split()

        expected_features = ["math_g1", "math_g2", "portuguese_g1", "portuguese_g2"]
        logistic = modeling.train_model(split)
        self.assertEqual(logistic.feature_names_in_.tolist(), expected_features)
        self.assertEqual(
            logistic.named_steps["standardscaler"].n_samples_seen_, len(split.train)
        )
        tree = modeling.train_model(split, "decision_tree")
        self.assertEqual(tree.feature_names_in_.tolist(), expected_features)
        self.assertEqual(tree.max_depth, 3)
        for model in (logistic, tree):
            validation = modeling.evaluate_validation(model, split)
            self.assertEqual(set(validation), {"accuracy", "f1", "confusion_matrix"})
            self.assertEqual(sum(map(sum, validation["confusion_matrix"])), len(split.validation))

    def test_baseline_writes_json_and_png(self):
        with TemporaryDirectory() as output_dir:
            with (
                patch.dict("os.environ", {"OUTPUT_DIR": output_dir}),
                patch.object(modeling, "read_students", return_value=student_rows()),
                patch("builtins.print"),
            ):
                report = modeling.run_baseline()
            metrics = Path(output_dir, "baseline_metrics.json").read_text(encoding="utf-8")
            self.assertEqual(json.loads(metrics), report)
            self.assertEqual(report["dummy_test"]["accuracy"], 0.5)
            self.assertEqual(report["dummy_test"]["f1"], 0)
            plot = Path(output_dir, "baseline_confusion_matrix.png").read_bytes()
            self.assertTrue(plot.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
