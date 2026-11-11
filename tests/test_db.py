"""P1 tests. They run on tiny made-up CSVs in tests/fixtures/raw, never the real data."""

import json
from pathlib import Path

import duckdb
import pytest

from claimssql.db import load_synpuf as ls
from claimssql.db import profile as pf
from claimssql.db.connection import get_connection

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("db") / "claims.duckdb"
    ls.build_database(FIXTURE_RAW, path)
    return path


@pytest.fixture()
def con(db_path):
    connection = get_connection(read_only=True, db_path=db_path)
    yield connection
    connection.close()


def _types(con, table) -> dict[str, str]:
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE {table}").fetchall()}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("DESYNPUF_ID", "VARCHAR"),
        ("CLM_ID", "VARCHAR"),
        ("PRVDR_NUM", "VARCHAR"),
        ("AT_PHYSN_NPI", "VARCHAR"),
        ("PRF_PHYSN_NPI_13", "VARCHAR"),
        ("ICD9_DGNS_CD_10", "VARCHAR"),
        ("ADMTNG_ICD9_DGNS_CD", "VARCHAR"),
        ("HCPCS_CD_45", "VARCHAR"),
        ("BENE_SEX_IDENT_CD", "VARCHAR"),
        ("SP_STATE_CODE", "VARCHAR"),
        ("BENE_COUNTY_CD", "VARCHAR"),
        ("TAX_NUM_3", "VARCHAR"),
        ("BENE_ESRD_IND", "VARCHAR"),
        ("CLM_FROM_DT", "BIGINT"),
        ("BENE_BIRTH_DT", "BIGINT"),
        ("PLAN_CVRG_MOS_NUM", "INTEGER"),
        ("CLM_PMT_AMT", None),
        ("SEGMENT", None),
        ("SP_DIABETES", None),  # chronic-condition flag, not a code column
        ("CLM_UTLZTN_DAY_CNT", None),
    ],
)
def test_column_type_rule(name, expected):
    assert ls.column_type(name) == expected


def test_row_counts_and_unions(con):
    counts = {t: (rows, cols) for t, rows, cols in ls.row_counts(con)}
    assert counts == {
        "beneficiary": (6, 10),  # 3 + 2 + 1 rows; 9 file columns + year
        "inpatient_claims": (4, 13),
        "outpatient_claims": (3, 11),
        "carrier_claims": (5, 10),  # 2 from 1A + 3 from 1B
    }


def test_year_column_comes_from_file_name(con):
    assert _types(con, "beneficiary")["year"] == "INTEGER"
    assert ls.beneficiary_ids_per_year(con) == [(2008, 3, 3), (2009, 2, 2), (2010, 1, 1)]


def test_codes_keep_leading_zeros(con):
    row = con.execute(
        "SELECT DESYNPUF_ID, SP_STATE_CODE, BENE_COUNTY_CD, BENE_SEX_IDENT_CD "
        "FROM beneficiary WHERE year = 2008 ORDER BY DESYNPUF_ID LIMIT 1"
    ).fetchone()
    assert row == ("000A", "01", "010", "1")

    row = con.execute(
        "SELECT CLM_ID, PRVDR_NUM, AT_PHYSN_NPI, CLM_DRG_CD, ICD9_PRCDR_CD_1 "
        "FROM inpatient_claims WHERE SEGMENT = 1 ORDER BY CLM_ID LIMIT 1"
    ).fetchone()
    assert row == ("000111", "0100AB", "0012345678", "061", "0061")

    tax = con.execute("SELECT DISTINCT TAX_NUM_1 FROM carrier_claims ORDER BY 1").fetchall()
    assert tax == [("012345678",), ("098765432",)]


def test_dates_stay_raw_integers_and_plan_months_is_integer(con):
    bene = _types(con, "beneficiary")
    assert bene["BENE_BIRTH_DT"] == "BIGINT"
    assert bene["BENE_DEATH_DT"] == "BIGINT"
    assert bene["PLAN_CVRG_MOS_NUM"] == "INTEGER"
    assert _types(con, "inpatient_claims")["CLM_FROM_DT"] == "BIGINT"
    assert con.execute("SELECT min(CLM_FROM_DT) FROM inpatient_claims").fetchone()[0] == 20080104
    # '00', '12', '06' in the file -> 0, 12, 6
    assert con.execute(
        "SELECT sum(PLAN_CVRG_MOS_NUM) FROM beneficiary WHERE year = 2008"
    ).fetchone()[0] == 18


def test_empty_fields_become_null(con):
    nulls = con.execute(
        "SELECT count(*) FROM beneficiary WHERE BENE_DEATH_DT IS NULL"
    ).fetchone()[0]
    assert nulls == 4
    # A column that is empty on every row still gets its forced type.
    assert _types(con, "inpatient_claims")["HCPCS_CD_1"] == "VARCHAR"


def test_multi_row_claim_ids(con):
    # CLM_ID 000111 has two segments in the inpatient fixture.
    assert ls.multi_row_claim_ids(con, "inpatient_claims") == (3, 1, 2)
    assert ls.multi_row_claim_ids(con, "outpatient_claims") == (3, 0, 1)
    checks = {(c, t, k): v for c, t, k, v in ls.run_checks(con)}
    assert checks[("clm_ids_with_multiple_rows", "inpatient_claims", "")] == 1
    assert checks[("distinct_desynpuf_id", "beneficiary", "2009")] == 2


def test_reload_replaces_instead_of_appending(tmp_path):
    path = tmp_path / "again.duckdb"
    ls.build_database(FIXTURE_RAW, path)
    ls.build_database(FIXTURE_RAW, path, tables=["carrier_claims"])
    con = get_connection(read_only=True, db_path=path)
    try:
        assert dict((t, r) for t, r, _ in ls.row_counts(con))["carrier_claims"] == 5
    finally:
        con.close()


def test_count_csv_rows_matches_loaded_rows(con):
    counts = {t: rows for t, rows, _ in ls.row_counts(con)}
    for table, files in ls.TABLE_SOURCES.items():
        assert sum(ls.count_csv_rows(FIXTURE_RAW / f) for f in files) == counts[table]


def test_missing_file_and_header_mismatch_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        ls.build_database(tmp_path, tmp_path / "x.duckdb", tables=["inpatient_claims"])

    for name in ls.TABLE_SOURCES["carrier_claims"]:
        (tmp_path / name).write_text((FIXTURE_RAW / name).read_text())
    b_file = tmp_path / ls.TABLE_SOURCES["carrier_claims"][1]
    b_file.write_text(b_file.read_text().replace('"HCPCS_CD_1"', '"HCPCS_CD_2"'))
    with pytest.raises(ValueError, match="header"):
        ls.build_database(tmp_path, tmp_path / "y.duckdb", tables=["carrier_claims"])


def test_read_only_connection_rejects_writes(con, tmp_path):
    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM beneficiary")
    with pytest.raises(FileNotFoundError):
        get_connection(read_only=True, db_path=tmp_path / "nope.duckdb")


def test_profile_numbers(con):
    profiles = {p.column: p for p in pf.profile_table(con, "inpatient_claims")}

    npi = profiles["AT_PHYSN_NPI"]  # 4 rows, 1 empty, 2 distinct values
    assert (npi.type, npi.null_pct, npi.distinct_count) == ("VARCHAR", 25.0, 2)
    assert (npi.min, npi.max) == ("0012345678", "9912345678")
    # Most frequent first: 0012345678 appears twice.
    assert npi.sample_values == "0012345678 | 9912345678"

    empty = profiles["HCPCS_CD_1"]  # empty on every row
    assert (empty.null_pct, empty.distinct_count, empty.min, empty.sample_values) == (
        100.0, 0, None, "",
    )  # fmt: skip

    assert profiles["CLM_FROM_DT"].min == "20080104"
    assert profiles["CLM_PMT_AMT"].distinct_count == 4


def test_low_cardinality_values_and_files(con, tmp_path):
    profiles = pf.profile_table(con, "beneficiary")
    low = pf.low_cardinality_values(con, profiles, max_distinct=2)
    cols = low["beneficiary"]
    assert cols["BENE_SEX_IDENT_CD"] == ["1", "2"]  # text, as loaded
    assert cols["BENE_ESRD_IND"] == ["0", "Y"]
    assert cols["BENE_DEATH_DT"] == [20080301, 20100615]  # numbers stay numbers
    assert "SP_STATE_CODE" not in cols  # 3 distinct values > threshold of 2
    assert "BENE_BIRTH_DT" not in cols
    assert "year" not in cols

    pf.write_profile(tmp_path / "column_profile.csv", profiles)
    lines = (tmp_path / "column_profile.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(pf.PROFILE_HEADER)
    assert len(lines) == 1 + 10
    json.dumps(low)  # values must be JSON-serialisable
