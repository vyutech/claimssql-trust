"""V2 (dry run) and V3 (self-consistency) tests on the tiny P1 fixture database."""

from pathlib import Path

import pytest

from claimssql.db import load_synpuf as ls
from claimssql.db.connection import get_connection
from claimssql.eval.compare import result_hash
from claimssql.verify import dry_run as v2
from claimssql.verify import self_consistency as v3

FIXTURES = Path(__file__).parent / "fixtures"
SLOW = "SELECT sum(a.range * b.range) FROM range(200000) a, range(200000) b"
COUNT_ALL = "SELECT count(*) FROM beneficiary"  # 6 rows in the fixture
COUNT_2008 = "SELECT count(*) FROM beneficiary WHERE year = 2008"  # 3
H6, H3 = result_hash([(6,)]), result_hash([(3,)])


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("p5") / "claims.duckdb"
    ls.build_database(FIXTURES / "raw", path)
    return path


@pytest.fixture()
def con(db_path):
    connection = get_connection(read_only=True, db_path=db_path)
    yield connection
    connection.close()


# -------------------------------------------------------------------- V2 ----


@pytest.mark.parametrize(
    "sql",
    [
        COUNT_ALL,
        "WITH c AS (SELECT year FROM beneficiary) SELECT count(*) FROM c",
        "SELECT CLM_ID FROM inpatient_claims UNION ALL SELECT CLM_ID FROM outpatient_claims",
        "SELECT CLM_ID FROM inpatient_claims EXCEPT SELECT CLM_ID FROM outpatient_claims",
        "(SELECT 1)",
        "-- a comment\nSELECT 1;",
    ],
)
def test_gate_accepts_queries(sql):
    assert v2.statement_gate(sql) is None


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE beneficiary",
        "CREATE TABLE t (x INTEGER)",
        "CREATE TABLE t AS SELECT * FROM beneficiary",
        "ALTER TABLE beneficiary ADD COLUMN x INTEGER",
        "INSERT INTO beneficiary SELECT * FROM beneficiary",
        "UPDATE beneficiary SET year = 2011",
        "DELETE FROM beneficiary",
        "TRUNCATE TABLE beneficiary",
        "COPY beneficiary TO 'out.csv'",
        "ATTACH 'other.duckdb' AS other",
        "PRAGMA database_list",
        "SET memory_limit = '1GB'",
        "SELECT 1; DROP TABLE beneficiary",
        "SELECT 1; SELECT 2",
    ],
)
def test_gate_rejects_everything_else(sql, con):
    assert v2.statement_gate(sql)[0] == v2.FORBIDDEN
    result = v2.dry_run(con, sql)
    assert (result.ok, result.error_class) == (False, v2.FORBIDDEN)
    assert result.result_hash is None and result.elapsed_ms == 0.0  # never executed
    assert con.execute(COUNT_ALL).fetchone()[0] == 6  # data untouched


@pytest.mark.parametrize("sql", ["SELEC 1", "SELECT FROM WHERE", "", "SELECT (1"])
def test_syntax_errors_are_parse_errors(sql, con):
    result = v2.dry_run(con, sql)
    assert (result.ok, result.error_class) == (False, v2.PARSE)
    assert result.error


def test_successful_run(con):
    result = v2.dry_run(con, COUNT_ALL)
    assert result.ok and result.error_class is None and result.error is None
    assert (result.row_count, result.truncated, result.result_hash) == (1, False, H6)
    assert result.elapsed_ms >= 0
    assert result.to_dict()["ok"] is True


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT PATIENT_AGE FROM beneficiary", v2.BINDER),  # unknown column
        ("SELECT b.year FROM beneficiary", v2.BINDER),  # undefined alias
        ("SELECT * FROM pharmacy_claims", v2.CATALOG),  # unknown table
        ("SELECT no_such_function(year) FROM beneficiary", v2.CATALOG),
        ("SELECT CAST(DESYNPUF_ID AS INTEGER) FROM beneficiary", v2.CONVERSION),  # '000A'
        ("SELECT CAST('2008-13-45' AS DATE)", v2.CONVERSION),
    ],
)
def test_execution_errors_are_classified(sql, expected, con):
    result = v2.dry_run(con, sql)
    assert not result.ok and result.error_class == expected, result.error
    assert result.result_hash is None


def test_timeout_on_a_deliberately_slow_query(con):
    result = v2.dry_run(con, SLOW, timeout_s=0.3)
    assert (result.ok, result.error_class) == (False, v2.TIMEOUT)
    assert result.elapsed_ms < 10_000
    assert v2.dry_run(con, COUNT_ALL).ok  # the connection still works afterwards


def test_truncated_result_runs_but_has_no_hash(con):
    result = v2.dry_run(con, "SELECT * FROM range(50)", max_rows=10)
    assert result.ok and result.truncated and result.row_count == 10
    assert result.result_hash is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ParserException: Parser Error: syntax error at or near x", v2.PARSE),
        ("BinderException: Binder Error: column not found", v2.BINDER),
        ("CatalogException: Catalog Error: Table does not exist", v2.CATALOG),
        ("ConversionException: Conversion Error: could not convert", v2.CONVERSION),
        ("timeout: query exceeded 30 s", v2.TIMEOUT),
        ("OutOfMemoryException: could not allocate", v2.OTHER),
        ("InvalidInputException: something odd: with a colon", v2.OTHER),
    ],
)
def test_classify_error(message, expected):
    assert v2.classify_error(message) == expected


# -------------------------------------------------------------------- V3 ----


def test_all_samples_agree_and_identical_sql_runs_once(con):
    result = v3.self_consistency(con, [COUNT_ALL] * 5, greedy_hash=H6)
    assert (result.k, result.agreement, result.cluster_sizes) == (5, 1.0, [5])
    assert result.majority_hash == H6 and result.greedy_in_majority
    assert result.distinct_sql_executed == 1
    assert result.outcomes == [H6] * 5


def test_different_sql_with_the_same_result_clusters_together(con):
    samples = [
        COUNT_ALL,
        "select COUNT(*)   from BENEFICIARY ;",  # same query, different spelling
        "SELECT count(*) /* everyone */ FROM beneficiary",
        "SELECT count(DESYNPUF_ID) AS n FROM beneficiary",  # different query, same result
        "SELECT sum(1) FROM beneficiary",
    ]
    result = v3.self_consistency(con, samples, greedy_hash=H6)
    assert result.agreement == 1.0 and result.cluster_sizes == [5]
    assert result.distinct_sql_executed == 3  # the first three normalise to one query


def test_split_clusters_and_greedy_in_the_minority(con):
    samples = [COUNT_ALL, COUNT_ALL, COUNT_ALL, COUNT_2008, COUNT_2008]
    majority = v3.self_consistency(con, samples, greedy_hash=H6)
    assert (majority.agreement, majority.cluster_sizes) == (0.6, [3, 2])
    assert majority.majority_hash == H6 and majority.greedy_in_majority

    minority = v3.self_consistency(con, samples, greedy_hash=H3)
    assert minority.majority_hash == H6 and not minority.greedy_in_majority


def test_failures_and_refusals_agree_with_nobody(con):
    samples = [COUNT_ALL, "SELECT nope FROM beneficiary", None, COUNT_ALL, "DROP TABLE beneficiary"]
    result = v3.self_consistency(con, samples, greedy_hash=H6)
    assert result.outcomes == [H6, v3.FAILED, v3.NO_SQL, H6, v3.FAILED]
    assert (result.agreement, result.cluster_sizes) == (0.4, [2])  # 2 of k=5, not 2 of 2
    assert result.greedy_in_majority
    assert con.execute(COUNT_ALL).fetchone()[0] == 6  # the DROP was never run


def test_no_usable_sample_gives_zero_agreement(con):
    result = v3.self_consistency(con, [None, "", "SELECT nope", None, "   "], greedy_hash=H6)
    assert (result.agreement, result.cluster_sizes, result.majority_hash) == (0.0, [], None)
    assert not result.greedy_in_majority
    assert result.distinct_sql_executed == 1


def test_tie_counts_as_majority_if_greedy_is_in_a_tied_cluster(con):
    samples = [COUNT_ALL, COUNT_ALL, COUNT_2008, COUNT_2008, None]
    in_tie = v3.self_consistency(con, samples, greedy_hash=H3)
    assert (in_tie.agreement, in_tie.cluster_sizes) == (0.4, [2, 2])
    assert in_tie.greedy_in_majority and in_tie.majority_hash == H3

    outside = v3.self_consistency(con, samples, greedy_hash=result_hash([(99,)]))
    assert not outside.greedy_in_majority and outside.majority_hash in {H6, H3}


def test_greedy_without_a_result_is_never_in_the_majority(con):
    result = v3.self_consistency(con, [COUNT_ALL] * 5, greedy_hash=None)
    assert result.agreement == 1.0 and not result.greedy_in_majority


def test_truncated_samples_do_not_cluster(con):
    result = v3.self_consistency(con, ["SELECT * FROM range(50)"] * 3 + [COUNT_ALL] * 2,
                                 greedy_hash=H6, max_rows=10)  # fmt: skip
    assert result.outcomes[:3] == [v3.TRUNCATED] * 3
    assert (result.agreement, result.cluster_sizes) == (0.4, [2])


def test_slow_sample_times_out_and_counts_as_failed(con):
    result = v3.self_consistency(con, [SLOW, COUNT_ALL], greedy_hash=H6, timeout_s=0.3)
    assert result.outcomes == [v3.FAILED, H6] and result.agreement == 0.5


def test_normalise_sql():
    same = [
        "SELECT count(*) FROM beneficiary WHERE year = 2008",
        "select COUNT(*)\n  from BENEFICIARY\n  where YEAR = 2008;",
        "SELECT count(*) FROM beneficiary -- people\nWHERE year = 2008",
    ]
    assert len({v3.normalise_sql(sql) for sql in same}) == 1
    assert v3.normalise_sql(same[0]) != v3.normalise_sql(COUNT_ALL)
    # Text values keep their case: 'a' and 'A' are different queries.
    assert v3.normalise_sql("SELECT 'a'") != v3.normalise_sql("SELECT 'A'")
    assert v3.normalise_sql("SELECT  (1 ;") == "SELECT (1"  # unparseable: tidied raw text
