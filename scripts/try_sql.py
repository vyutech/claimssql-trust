"""Run one query against the claims database (read-only) and show the result.

Prints the first rows as a table, then the row count and the time taken. Meant
for trying out gold SQL while writing benchmark questions.

Usage:
    python scripts/try_sql.py "SELECT count(*) FROM beneficiary"
    python scripts/try_sql.py -f query.sql
    python scripts/try_sql.py -f query.sql --rows 50
    python scripts/try_sql.py "SELECT * FROM carrier_claims LIMIT 2" --vertical
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from claimssql.db.connection import DEFAULT_DB_PATH, get_connection
from claimssql.eval.compare import MAX_ROWS, execute_sql

MAX_CELL_WIDTH = 40


def _cell(value: object) -> str:
    text = "NULL" if value is None else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= MAX_CELL_WIDTH else text[: MAX_CELL_WIDTH - 1] + "…"


def format_table(columns: list[str], rows: list[tuple]) -> str:
    cells = [[_cell(v) for v in row] for row in rows]
    widths = [
        max(len(name), *(len(row[i]) for row in cells)) if cells else len(name)
        for i, name in enumerate(columns)
    ]
    # Numbers line up better on the right; everything else on the left.
    numeric = [
        bool(rows) and all(isinstance(r[i], (int, float)) or r[i] is None for r in rows)
        for i in range(len(columns))
    ]

    def line(values: list[str]) -> str:
        parts = [v.rjust(w) if numeric[i] else v.ljust(w)
                 for i, (v, w) in enumerate(zip(values, widths, strict=True))]  # fmt: skip
        return " | ".join(parts).rstrip()

    rule = "-+-".join("-" * w for w in widths)
    return "\n".join([line(columns), rule, *(line(row) for row in cells)])


def format_vertical(columns: list[str], rows: list[tuple]) -> str:
    """One "column: value" line per cell, rows separated by a header line.

    For tables too wide to read across (carrier_claims has 142 columns).
    Values are printed in full, not shortened as in the table view.
    """
    blocks = []
    for number, row in enumerate(rows, start=1):
        lines = [f"--- row {number} ---"]
        for name, value in zip(columns, row, strict=True):
            text = "NULL" if value is None else str(value).replace("\n", " ")
            lines.append(f"{name}: {text}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("sql", nargs="?", help="the query, in quotes")
    source.add_argument("-f", "--file", type=Path, help="read the query from this file")
    parser.add_argument("--rows", type=int, default=20, help="rows to show (default 20)")
    parser.add_argument("--vertical", action="store_true",
                        help='print each row as "column: value" lines (for wide tables)')
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds (default 60)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    sql = args.file.read_text(encoding="utf-8") if args.file else args.sql
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    con = get_connection(read_only=True, db_path=args.db_path)
    try:
        result = execute_sql(con, sql, timeout_s=args.timeout, max_rows=MAX_ROWS)
        if not result.ok:
            print(f"ERROR: {result.error}", file=sys.stderr)
            return 1
        count = f"{len(result.rows):,}"
        if result.truncated:
            # More rows than the scoring cap: count them all with a second query.
            inner = sql.strip().rstrip(";")
            total = execute_sql(con, f"SELECT count(*) FROM ({inner})", timeout_s=args.timeout)
            count = f"{total.rows[0][0]:,}" if total.ok else f"more than {MAX_ROWS:,}"
    finally:
        con.close()

    formatter = format_vertical if args.vertical else format_table
    print(formatter(result.columns, result.rows[: args.rows]))
    shown = min(args.rows, len(result.rows))
    print(f"\n{count} row(s); showing {shown}. {result.elapsed_ms:,.0f} ms")
    if result.truncated:
        print(f"WARNING: over the {MAX_ROWS:,}-row limit for benchmark results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
