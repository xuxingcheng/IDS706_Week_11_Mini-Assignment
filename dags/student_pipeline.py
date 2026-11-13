"""Daily ingestion, parallel cleaning, SQLite load, and baseline analysis."""

from datetime import timedelta

import pendulum
from airflow.sdk import dag, task, task_group


@dag(
    dag_id="student_performance_pipeline",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=4,
    is_paused_upon_creation=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["assignment", "students"],
)
def student_pipeline():
    @task
    def ingest(subject: str) -> str:
        from pipeline import ingest as ingest_table

        return ingest_table(subject)

    @task
    def clean(path: str) -> str:
        from pipeline import clean as clean_table

        return clean_table(path)

    @task_group
    def prepare_subject(subject: str):
        return clean(ingest(subject))

    @task
    def merge_and_load(math_path: str, portuguese_path: str) -> dict:
        from pipeline import merge_and_load as load_tables

        return load_tables(math_path, portuguese_path)

    @task
    def analyze(summary: dict) -> dict:
        from modeling import run_baseline

        print(f"Reading {summary['matched_rows']} cleaned students from SQLite")
        return run_baseline()

    # Neither subject branch depends on the other, so LocalExecutor can overlap them.
    math = prepare_subject.override(group_id="math")("mat")
    portuguese = prepare_subject.override(group_id="portuguese")("por")
    analyze(merge_and_load(math, portuguese))


student_pipeline()
