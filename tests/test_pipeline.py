import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name) / "data"
        self.output_dir = Path(temporary.name) / "outputs"
        settings = patch.multiple(
            pipeline,
            DATA_DIR=self.data_dir,
            DB_PATH=self.data_dir / "students.db",
            OUTPUT_DIR=self.output_dir,
        )
        settings.start()
        self.addCleanup(settings.stop)

    def clean_rows(self, subject, rows):
        path = self.data_dir / f"student-{subject}.csv"
        path.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(path, sep=";", index=False)
        return pipeline.clean(str(path))

    def student(self, **overrides):
        return {
            "school": "GP", "sex": "F", "age": 16, "address": "U",
            "famsize": "LE3", "Pstatus": "T", "Medu": 2, "Fedu": 2,
            "Mjob": "teacher", "Fjob": "services", "reason": "course",
            "nursery": "yes", "internet": "no",
            "G1": 12, "G2": 10, "G3": 10, "absences": 0,
            **overrides,
        }

    def test_clean_trims_and_drops_invalid_required_values_and_exact_duplicates(self):
        with self.assertLogs(pipeline.LOGGER, level="INFO") as logs:
            path = self.clean_rows("mat", [
                self.student(school=" GP ", G1=" 12 "),
                self.student(),
                self.student(absences=None),
                self.student(school=" "),
                self.student(G2=None),
                self.student(G1="invalid"),
                self.student(G3=21),
                self.student(G3=9.5),
                self.student(age=23),
            ])

        clean = pd.read_csv(path)
        self.assertEqual(clean["school"].tolist(), ["GP", "GP"])
        self.assertEqual(clean["G1"].tolist(), [12, 12])
        self.assertEqual(clean["absences"].isna().sum(), 1)
        self.assertIn("6 invalid removed, 1 exact duplicates removed, 2 clean", logs.output[0])

    def test_load_excludes_ambiguous_keys_and_can_replace_a_previous_load(self):
        math_rows = [
            self.student(age=16, G3=10),
            self.student(age=17, G3=9),
            self.student(age=18),
            self.student(age=19),
        ]
        portuguese = self.clean_rows("por", [
            self.student(age=16, G1=15, G2=16),
            self.student(age=17),
            self.student(age=18),
            self.student(age=18, absences=3),
            self.student(age=20),
        ])

        summary = pipeline.merge_and_load(self.clean_rows("mat", math_rows), portuguese)
        self.assertEqual(summary["ambiguous_key_groups"], 1)
        self.assertEqual(summary["math_ambiguous_rows"], 1)
        self.assertEqual(summary["portuguese_ambiguous_rows"], 2)
        self.assertEqual(summary["math_unmatched_rows"], 1)
        self.assertEqual(summary["portuguese_unmatched_rows"], 1)
        self.assertEqual(summary["matched_rows"], 2)
        self.assertEqual(summary["passed_math"], 1)
        self.assertEqual(summary["failed_math"], 1)
        self.assertEqual(
            json.loads((self.output_dir / "data_quality.json").read_text()), summary
        )

        published = pipeline.read_students()
        self.assertTrue(published["student_id"].is_unique)
        self.assertEqual(
            published.sort_values("math_final")["passed_math"].tolist(), [0, 1]
        )
        self.assertEqual(
            published.loc[published["math_final"].eq(10), "portuguese_g1"].item(), 15
        )
        pipeline.merge_and_load(self.clean_rows("mat", math_rows[::-1]), portuguese)
        pd.testing.assert_frame_equal(pipeline.read_students(), published)

        with self.assertRaisesRegex(ValueError, "No students matched"):
            pipeline.merge_and_load(self.clean_rows("mat", [self.student(age=22)]), portuguese)
        pd.testing.assert_frame_equal(pipeline.read_students(), published)


if __name__ == "__main__":
    unittest.main()
