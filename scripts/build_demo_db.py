"""Build the files the deployed demo needs.

  data/demo.duckdb           5,000 fixed random beneficiaries with all their claims
  app_data/explorer.jsonl    what the Explorer page shows, from a finished run
  app_data/results.json      numbers and chart data for Home and Results (results/tables)
  app_data/samples.json      one-click sample questions (dev split of the benchmark)
  app_data/examples.json     real cases from the run for the How-it-works page

Usage:
    python scripts/build_demo_db.py
    python scripts/build_demo_db.py --run-id full --n 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from claimssql.config import REPO_ROOT, RESULTS_RUNS_DIR, RESULTS_TABLES_DIR
from claimssql.db.connection import DEFAULT_DB_PATH, get_connection
from claimssql.db.demo import DEFAULT_N, DEFAULT_SALT, build_demo_db
from claimssql.eval.explorer import (
    build_explorer_rows,
    check_examples,
    results_detail,
    results_summary,
    sample_questions,
    write_explorer,
)
from claimssql.eval.run_benchmark import read_run
from claimssql.schema.catalogue import load_catalogue, validate_against_db

DEFAULT_DEMO_DB = REPO_ROOT / "data" / "demo.duckdb"
DEFAULT_EXPLORER = REPO_ROOT / "app_data" / "explorer.jsonl"
DEFAULT_QUESTIONS = REPO_ROOT / "benchmark" / "questions.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_DEMO_DB)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--run-id", default="full", help="run file for the explorer")
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_RUNS_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--explorer-out", type=Path, default=DEFAULT_EXPLORER)
    args = parser.parse_args(argv)

    counts = build_demo_db(args.source, args.out, args.n, args.salt)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    for table, rows in counts.items():
        print(f"  {table:<18} {rows:>9,} rows")

    con = get_connection(read_only=True, db_path=args.out)
    try:
        report = validate_against_db(load_catalogue(), con)
    finally:
        con.close()
    if report.has_mismatches:
        print("ERROR: demo database does not match the catalogue:", file=sys.stderr)
        for line in report.lines():
            print(f"  {line}", file=sys.stderr)
        return 1
    print("  matches the catalogue")

    records = read_run(args.runs_dir / f"{args.run_id}.jsonl")
    if not records:
        print(f"No run file {args.run_id}.jsonl: explorer data not written.", file=sys.stderr)
        return 1
    questions = {}
    for line in args.questions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record
    # Gold and answer previews come from the FULL database, as in the benchmark.
    con = get_connection(read_only=True, db_path=args.source)
    try:
        rows = build_explorer_rows(records, questions, con)
    finally:
        con.close()
    write_explorer(args.explorer_out, rows)
    app_data = args.explorer_out.parent
    results = {**results_summary(RESULTS_TABLES_DIR), **results_detail(RESULTS_TABLES_DIR)}
    (app_data / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (app_data / "examples.json").write_text(
        json.dumps(check_examples(rows), indent=1, ensure_ascii=False), encoding="utf-8")
    (app_data / "samples.json").write_text(
        json.dumps(sample_questions(list(questions.values())), indent=1, ensure_ascii=False),
        encoding="utf-8")  # fmt: skip
    # The app draws its own charts from results.json; the greyscale PNGs in
    # results/figures/ are only for the printed report. Remove an old copy.
    (app_data / "fig7_2_harmful_by_config.png").unlink(missing_ok=True)
    models = sorted({row["model_key"] for row in rows})
    print(f"Wrote {args.explorer_out} ({args.explorer_out.stat().st_size / 1e6:.1f} MB, "
          f"{len(rows)} rows, models: {', '.join(models)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
