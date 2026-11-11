import datetime as dt
from decimal import Decimal

import duckdb
import pytest

from claimssql.eval.compare import (
    NULL_SENTINEL,
    execute_sql,
    normalise_result,
    result_hash,
)


@pytest.fixture()
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


def test_row_order_does_not_matter():
    assert result_hash([(1, "a"), (2, "b")]) == result_hash([(2, "b"), (1, "a")])


def test_duplicate_rows_count():
    assert result_hash([(1,), (1,)]) != result_hash([(1,)])


def test_column_order_matters():
    assert result_hash([(1, "a")]) != result_hash([("a", 1)])


def test_floats_rounded_to_4_dp():
    assert result_hash([(1.00004,)]) == result_hash([(1.00001,)])  # both 1.0
    assert result_hash([(1.0001,)]) != result_hash([(1.0002,)])
    assert result_hash([(0.1 + 0.2,)]) == result_hash([(0.3,)])
    assert normalise_result([(-0.00001,)]) == [[0.0]]  # no negative zero


def test_decimal_int_and_float_are_one_number():
    assert result_hash([(Decimal("2.50"),)]) == result_hash([(2.5,)])
    assert result_hash([(5,)]) == result_hash([(5.0,)]) == result_hash([(Decimal("5"),)])


def test_null_is_a_sentinel_not_a_string_or_zero():
    assert normalise_result([(None,)]) == [[NULL_SENTINEL]]
    assert result_hash([(None,)]) != result_hash([("",)])
    assert result_hash([(None,)]) != result_hash([("NULL",)])
    assert result_hash([(None,)]) != result_hash([(0,)])
    assert result_hash([(None, 1)]) == result_hash([(None, 1)])


def test_strings_stripped_but_case_kept():
    assert result_hash([("  CA ",)]) == result_hash([("CA",)])
    assert result_hash([("ca",)]) != result_hash([("CA",)])
    assert result_hash([("1",)]) != result_hash([(1,)])  # text '1' is not the number 1


def test_other_types():
    assert result_hash([(True,)]) != result_hash([(1,)])
    assert normalise_result([(dt.date(2008, 1, 4),)]) == [["2008-01-04"]]
    assert result_hash([(float("nan"),)]) == result_hash([(float("nan"),)])
    assert result_hash([(float("nan"),)]) != result_hash([(None,)])
    assert normalise_result([(2**60,)]) == [[2**60]]  # too big for a float: kept exact


def test_mixed_types_in_a_column_still_sort():
    assert result_hash([(1,), ("a",), (None,)]) == result_hash([(None,), ("a",), (1,)])


def test_empty_result_has_a_stable_hash():
    assert result_hash([]) == result_hash([])
    assert result_hash([]) != result_hash([(None,)])


def test_column_aliases_do_not_matter(con):
    a = execute_sql(con, "SELECT 1 AS n, 'x' AS label")
    b = execute_sql(con, "SELECT 1 AS total, 'x' AS name")
    assert a.columns == ["n", "label"] and b.columns == ["total", "name"]
    assert result_hash(a.rows) == result_hash(b.rows)


def test_execute_returns_rows_and_timing(con):
    result = execute_sql(con, "SELECT * FROM (VALUES (2, 'b'), (1, 'a')) t(n, s) ORDER BY n")
    assert result.ok and result.error is None
    assert result.rows == [(1, "a"), (2, "b")]
    assert result.truncated is False
    assert result.elapsed_ms >= 0


def test_sql_decimal_and_double_compare_equal(con):
    a = execute_sql(con, "SELECT CAST(10.5 AS DECIMAL(10,2)), count(*) FROM range(3)")
    b = execute_sql(con, "SELECT CAST(10.5 AS DOUBLE), CAST(3 AS DOUBLE)")
    assert result_hash(a.rows) == result_hash(b.rows)


def test_truncation(con):
    result = execute_sql(con, "SELECT * FROM range(25)", max_rows=10)
    assert len(result.rows) == 10 and result.truncated is True
    exact = execute_sql(con, "SELECT * FROM range(10)", max_rows=10)
    assert len(exact.rows) == 10 and exact.truncated is False


def test_errors_are_returned_not_raised(con):
    result = execute_sql(con, "SELECT nope FROM missing_table")
    assert not result.ok
    assert "missing_table" in result.error
    assert result.rows == [] and result.columns == []
    assert not execute_sql(con, "SELEC 1").ok


def test_timeout_interrupts_the_query(con):
    slow = "SELECT sum(a.range * b.range) FROM range(200000) a, range(200000) b"
    result = execute_sql(con, slow, timeout_s=0.3)
    assert result.error is not None and result.error.startswith("timeout")
    assert result.elapsed_ms < 10_000
    # The connection is still usable afterwards.
    assert execute_sql(con, "SELECT 1").rows == [(1,)]
