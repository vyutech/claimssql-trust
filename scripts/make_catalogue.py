"""Draft schema/catalogue.yaml from the live DB, the column profile and the CMS codebook.

Never overwrites: if the output file exists the script stops. To compare a fresh
draft with a reviewed catalogue, write it elsewhere with --out.

Usage:
    python scripts/make_catalogue.py
    python scripts/make_catalogue.py --out schema/catalogue.draft.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from claimssql.config import REPO_ROOT, RESULTS_TABLES_DIR
from claimssql.db.connection import DEFAULT_DB_PATH, get_connection
from claimssql.schema.catalogue import DEFAULT_CATALOGUE_PATH
from claimssql.schema.codebook import extract_text, index_by_column, parse_codebook
from claimssql.schema.skeleton import build_catalogue, read_profile, write_catalogue

DEFAULT_CODEBOOK = REPO_ROOT / "docs" / "de-10-codebook.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--profile", type=Path, default=RESULTS_TABLES_DIR / "column_profile.csv")
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_CATALOGUE_PATH)
    parser.add_argument("--max-coded-distinct", type=int, default=100,
                        help="only columns with at most this many distinct values get codes")
    args = parser.parse_args()

    if args.out.exists():
        print(f"{args.out} already exists - not overwriting. Use --out to write a draft "
              "somewhere else.", file=sys.stderr)
        return 1
    for path in (args.profile, args.codebook):
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 1

    codebook = index_by_column(parse_codebook(extract_text(args.codebook)))
    con = get_connection(read_only=True, db_path=args.db_path)
    try:
        catalogue, report = build_catalogue(
            con, read_profile(args.profile), codebook, args.max_coded_distinct
        )
    finally:
        con.close()
    write_catalogue(catalogue, args.out)

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Wrote {args.out}")
    for table in catalogue.tables:
        n_columns = sum(len(e.members()) for e in table.columns)
        n_coded = sum(bool(e.codes) for e in table.columns)
        print(f"  {table.name:<18} {len(table.columns):>3} entries covering "
              f"{n_columns:>3} columns, {n_coded} with codes")
    print(f"  total: {catalogue.n_entries()} entries")

    print(f"\nExcluded (include: false): {len(report.excluded)} entries")
    for table, name, reason in report.excluded:
        print(f"  {table}.{name} - {reason}")
    print(f"\nNot described by the codebook: {len(report.undescribed)}")
    for table, name in report.undescribed:
        print(f"  {table}.{name}")
    print(f"\nValues in the data with no codebook meaning: {len(report.values_without_code)}")
    for table, name, values in report.values_without_code:
        print(f"  {table}.{name}: {', '.join(values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
