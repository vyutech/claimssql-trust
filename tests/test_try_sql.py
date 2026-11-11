"""scripts/try_sql.py, run as a real command against the fixture database."""

import subprocess
import sys
from pathlib import Path

import pytest

from claimssql.db import load_synpuf as ls

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "try_sql.py"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("try") / "claims.duckdb"
    ls.build_database(ROOT / "tests" / "fixtures" / "raw", path)
    return path


def run(db_path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db-path", str(db_path), *args],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )  # fmt: skip


def test_query_from_argument(db_path):
    sql = "SELECT DESYNPUF_ID, BENE_DEATH_DT, year FROM beneficiary ORDER BY 3, 1"
    result = run(db_path, sql)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].split(" | ") == ["DESYNPUF_ID", "BENE_DEATH_DT", "year"]
    assert set(lines[1]) == {"-", "+"}
    assert lines[2].startswith("000A") and "NULL" in lines[2] and lines[2].endswith("2008")
    assert "6 row(s); showing 6." in result.stdout and " ms" in result.stdout


def test_query_from_file_and_row_limit(db_path, tmp_path):
    query = tmp_path / "query.sql"
    query.write_text("SELECT *\nFROM range(50);\n", encoding="utf-8")
    result = run(db_path, "-f", str(query), "--rows", "5")
    assert result.returncode == 0, result.stderr
    assert "50 row(s); showing 5." in result.stdout
    assert len(result.stdout.splitlines()) == 2 + 5 + 2  # header, rule, rows, blank, summary


def test_errors_and_read_only(db_path):
    bad = run(db_path, "SELECT nope FROM beneficiary")
    assert bad.returncode == 1 and "ERROR" in bad.stderr
    write = run(db_path, "DELETE FROM beneficiary")
    assert write.returncode == 1 and "ERROR" in write.stderr
    assert "6 row(s)" in run(db_path, "SELECT * FROM beneficiary").stdout


def test_more_rows_than_the_benchmark_limit(db_path):
    result = run(db_path, "SELECT * FROM range(10500)")
    assert result.returncode == 0, result.stderr
    assert "10,500 row(s)" in result.stdout and "WARNING" in result.stdout
