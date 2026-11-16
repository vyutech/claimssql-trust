"""Precompute what the app's "Benchmark explorer" shows, from a finished run.

The deployed demo cannot ship the 900 MB claims database, so everything the
explorer displays is computed here, on the laptop, and written to one small
file (app_data/explorer.jsonl): per question x model, the C0 and C4 decisions,
the verifier details, and the first rows of the gold result and of each answer.

C4 uses each model's tau chosen on the dev split, exactly as in the report.
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from claimssql.eval.compare import execute_sql
from claimssql.eval.score import score_record, verifier_outputs

PREVIEW_ROWS = 10
EXPLORER_CONFIGS = ("C0", "C4")


def jsonable(value: Any) -> Any:
    """A cell value as plain JSON (numbers, text, null)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


class _Runner:
    """Runs preview queries once each (many answers repeat the same SQL)."""

    def __init__(self, con: duckdb.DuckDBPyConnection, timeout_s: float) -> None:
        self.con = con
        self.timeout_s = timeout_s
        self._seen: dict[str, dict[str, Any]] = {}

    def preview(self, sql: str | None) -> dict[str, Any] | None:
        if not sql:
            return None
        if sql not in self._seen:
            result = execute_sql(self.con, sql, timeout_s=self.timeout_s, max_rows=PREVIEW_ROWS)
            self._seen[sql] = {
                "columns": result.columns,
                "rows": [[jsonable(v) for v in row] for row in result.rows],
                "more_rows": result.truncated,
                "error": result.error,
            }
        return self._seen[sql]


def _verification(record: dict[str, Any], config: str) -> dict[str, Any]:
    v1, v2 = verifier_outputs(record, config)
    v3 = record.get("v3") or {}
    return {
        "unknown_tables": (v1 or {}).get("unknown_tables", []),
        "unknown_columns": (v1 or {}).get("unknown_columns", []),
        "parse_error": (v1 or {}).get("parse_error"),
        "exec_error_class": (v2 or {}).get("error_class"),
        "exec_error": (v2 or {}).get("error"),
        # V3 is only used by C4; C0 has no self-consistency check.
        "agreement": v3.get("agreement") if config == "C4" else None,
        "cluster_sizes": v3.get("cluster_sizes") if config == "C4" else None,
        "k": v3.get("k") if config == "C4" else None,
    }


def build_explorer_rows(
    records: list[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    con: duckdb.DuckDBPyConnection,
    timeout_s: float = 60.0,
) -> list[dict[str, Any]]:
    """One explorer row per record (question x model), in run-file order."""
    # Imported here: the deployed app reads explorer files but has no scikit-learn.
    from claimssql.eval import metrics

    taus = {}
    for model in {r["model_key"] for r in records}:
        dev = [r for r in records if r["model_key"] == model and r.get("split") == "dev"]
        taus[model] = metrics.choose_tau(dev)[0]

    runner = _Runner(con, timeout_s)
    rows = []
    for record in records:
        question = questions.get(record["question_id"], {})
        tau = taus[record["model_key"]]
        configs = {}
        for config in EXPLORER_CONFIGS:
            case = score_record(record, config, tau)
            configs[config] = {
                "answered": case.answered,
                "reason_code": case.reason_code,
                "correct": case.correct,
                "region": case.region,
                "sql": case.sql,
                # The SQL the model wrote even when the system abstained, for display.
                "model_sql": _model_sql(record, config),
                "result": runner.preview(case.sql),
                "verification": _verification(record, config),
            }
        rows.append({
            "model_key": record["model_key"],
            "model": record["model"],
            "question_id": record["question_id"],
            "question": record["question"],
            "group": record["group"],
            "split": record.get("split"),
            "answerable": record["answerable"],
            "notes": question.get("notes", ""),
            "gold_sql": question.get("gold_sql"),
            "gold": runner.preview(question.get("gold_sql")),
            "tau": tau,
            "configs": configs,
        })  # fmt: skip
    return rows


def _model_sql(record: dict[str, Any], config: str) -> str | None:
    generation = record["forced"] if config == "C0" else record["abstain"]
    parsed = generation.get("parsed") or {}
    return parsed.get("sql")


def write_explorer(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_explorer(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]  # fmt: skip
