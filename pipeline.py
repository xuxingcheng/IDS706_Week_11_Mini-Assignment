"""Ingest, clean, and link the UCI Student Performance subject files."""

import hashlib
import json
import logging
import os
import sqlite3
from contextlib import closing
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "students.db"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))
SOURCE_URL = "https://archive.ics.uci.edu/static/public/320/student%2Bperformance.zip"
LOGGER = logging.getLogger(__name__)

# These fields come from student-merge.R in the UCI source archive.
MATCH_KEYS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
]
NUMERIC_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}


def ingest(subject: str) -> str:
    """Download and save the raw CSV for Math (mat) or Portuguese (por)."""
    if subject not in {"mat", "por"}:
        raise ValueError(f"Unknown subject: {subject!r}. Use 'mat' or 'por'.")

    filename = f"student-{subject}.csv"
    with urlopen(SOURCE_URL, timeout=30) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as outer:
        with ZipFile(BytesIO(outer.read("student.zip"))) as inner:
            contents = inner.read(filename)

    path = DATA_DIR / "raw" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return str(path)


def clean(path: str) -> str:
    """Normalize fields and remove rows with invalid required values or exact duplicates."""
    source = Path(path)
    frame = pd.read_csv(source, sep=";", dtype="string")
    frame.columns = frame.columns.str.strip()
    required = MATCH_KEYS + ["G1", "G2", "G3"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{source.name} is missing columns: {', '.join(sorted(missing))}")

    frame = frame.apply(lambda column: column.str.strip().replace("", pd.NA))
    valid = frame[required].notna().all(axis=1)
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        valid &= values.between(minimum, maximum) & values.mod(1).eq(0)
        frame[column] = values

    source_rows = len(frame)
    frame = frame.loc[valid].copy()
    valid_rows = len(frame)
    for column in NUMERIC_RANGES:
        frame[column] = frame[column].astype(int)
    frame = frame.drop_duplicates()
    LOGGER.info(
        "%s: %d source, %d invalid removed, %d exact duplicates removed, %d clean",
        source.name, source_rows, source_rows - valid_rows, valid_rows - len(frame), len(frame),
    )

    destination = DATA_DIR / "clean" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return str(destination)


def merge_and_load(math_path: str, portuguese_path: str) -> dict:
    """Link unambiguous students and replace the SQLite table in one transaction."""
    math = pd.read_csv(math_path)
    portuguese = pd.read_csv(portuguese_path)
    ambiguous_keys = pd.concat([
        math.loc[math.duplicated(MATCH_KEYS, keep=False), MATCH_KEYS],
        portuguese.loc[portuguese.duplicated(MATCH_KEYS, keep=False), MATCH_KEYS],
    ]).drop_duplicates()
    unambiguous = []
    for subject in (math, portuguese):
        tagged = subject.merge(ambiguous_keys, on=MATCH_KEYS, how="left", indicator=True)
        unambiguous.append(tagged.loc[tagged["_merge"].eq("left_only")])
    math_unique, portuguese_unique = unambiguous

    math_grades = {"G1": "math_g1", "G2": "math_g2", "G3": "math_final"}
    portuguese_grades = {"G1": "portuguese_g1", "G2": "portuguese_g2"}
    linked = math_unique[MATCH_KEYS + list(math_grades)].rename(columns=math_grades).merge(
        portuguese_unique[MATCH_KEYS + list(portuguese_grades)].rename(columns=portuguese_grades),
        on=MATCH_KEYS,
        validate="one_to_one",
    )
    if linked.empty:
        raise ValueError("No students matched on the 13 UCI fields after excluding ambiguous keys.")

    students = linked[
        ["math_g1", "math_g2", "portuguese_g1", "portuguese_g2", "math_final"]
    ].copy()
    students.insert(0, "student_id", [
        hashlib.sha256(json.dumps(key).encode()).hexdigest()
        for key in linked[MATCH_KEYS].itertuples(index=False, name=None)
    ])
    students["passed_math"] = students["math_final"].ge(10).astype(int)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TABLE IF EXISTS students")
        connection.execute("""
            CREATE TABLE students (
                student_id TEXT PRIMARY KEY NOT NULL,
                math_g1 INTEGER NOT NULL,
                math_g2 INTEGER NOT NULL,
                portuguese_g1 INTEGER NOT NULL,
                portuguese_g2 INTEGER NOT NULL,
                math_final INTEGER NOT NULL,
                passed_math INTEGER NOT NULL
            )
        """)
        connection.executemany(
            "INSERT INTO students VALUES (?, ?, ?, ?, ?, ?, ?)",
            students.itertuples(index=False, name=None),
        )

    passed = int(students["passed_math"].sum())
    summary = {
        "source_url": SOURCE_URL,
        "math_clean_rows": len(math),
        "portuguese_clean_rows": len(portuguese),
        "ambiguous_key_groups": len(ambiguous_keys),
        "math_ambiguous_rows": len(math) - len(math_unique),
        "portuguese_ambiguous_rows": len(portuguese) - len(portuguese_unique),
        "math_unmatched_rows": len(math_unique) - len(students),
        "portuguese_unmatched_rows": len(portuguese_unique) - len(students),
        "matched_rows": len(students),
        "passed_math": passed,
        "failed_math": len(students) - passed,
        "database": DB_PATH.name,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "data_quality.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Student load complete: %s", summary)
    return summary


def read_students() -> pd.DataFrame:
    """Read the published student rows for model training and the dashboard."""
    with closing(sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        return pd.read_sql_query("SELECT * FROM students ORDER BY student_id", connection)
