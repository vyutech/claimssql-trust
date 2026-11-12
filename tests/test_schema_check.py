"""V1 tests on hand-made good and bad SQL. Nothing is executed."""

from pathlib import Path

import pytest

from claimssql.schema.catalogue import load_catalogue
from claimssql.verify.schema_check import check_sql, schema_from_catalogue

SCHEMA = {
    "beneficiary": {
        "DESYNPUF_ID", "BENE_BIRTH_DT", "BENE_DEATH_DT", "BENE_SEX_IDENT_CD", "SP_STATE_CODE",
        "SP_DIABETES", "MEDREIMB_IP", "YEAR",
    },
    "inpatient_claims": {
        "DESYNPUF_ID", "CLM_ID", "SEGMENT", "CLM_FROM_DT", "CLM_THRU_DT", "CLM_PMT_AMT",
        "PRVDR_NUM", "CLM_DRG_CD", "CLM_UTLZTN_DAY_CNT", "ICD9_DGNS_CD_1", "ICD9_DGNS_CD_2",
    },
    "outpatient_claims": {
        "DESYNPUF_ID", "CLM_ID", "SEGMENT", "CLM_FROM_DT", "CLM_PMT_AMT", "ICD9_DGNS_CD_1",
    },
    "carrier_claims": {
        "DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "HCPCS_CD_1", "LINE_NCH_PMT_AMT_1",
        "LINE_NCH_PMT_AMT_2",
    },
}  # fmt: skip

VALID = {
    "simple": "SELECT DESYNPUF_ID, year FROM beneficiary WHERE year = 2008",
    "case_insensitive": "select desynpuf_id, Year from BENEFICIARY where SP_diabetes = 1",
    "quoted_identifiers": """
        SELECT "DESYNPUF_ID", "year" FROM "beneficiary" WHERE "SP_STATE_CODE" = '05'""",
    "table_alias": "SELECT b.DESYNPUF_ID FROM beneficiary AS b WHERE b.year = 2009",
    "full_table_name_as_qualifier": "SELECT beneficiary.year FROM beneficiary",
    "join_with_aliases": """
        SELECT b.SP_STATE_CODE, SUM(i.CLM_PMT_AMT)
        FROM beneficiary b JOIN inpatient_claims i ON i.DESYNPUF_ID = b.DESYNPUF_ID
        WHERE b.year = 2008 GROUP BY b.SP_STATE_CODE""",
    "join_using": "SELECT year FROM beneficiary JOIN inpatient_claims USING (DESYNPUF_ID)",
    "select_alias_in_group_order_having": """
        SELECT SP_STATE_CODE AS state, COUNT(*) AS n FROM beneficiary
        GROUP BY state HAVING n > 10 ORDER BY n DESC, state""",
    "select_alias_in_where": "SELECT CLM_PMT_AMT * 2 AS dbl FROM inpatient_claims WHERE dbl > 10",
    "group_by_position_and_all": "SELECT year, COUNT(*) FROM beneficiary GROUP BY 1 ORDER BY ALL",
    "cte_output_columns": """
        WITH per_claim AS (
          SELECT CLM_ID, SUM(CLM_PMT_AMT) AS paid, MIN(CLM_FROM_DT) AS start_dt
          FROM inpatient_claims GROUP BY CLM_ID)
        SELECT AVG(paid), MIN(start_dt) FROM per_claim WHERE per_claim.paid > 0""",
    "cte_chain": """
        WITH s AS (SELECT CLM_ID, SUM(CLM_PMT_AMT) AS paid FROM inpatient_claims GROUP BY 1),
             h AS (SELECT CLM_ID, paid, paid * 2 AS doubled FROM s)
        SELECT COUNT(*) FROM h WHERE doubled > (SELECT AVG(paid) FROM s)""",
    "cte_with_column_list": """
        WITH c(person, total) AS (SELECT DESYNPUF_ID, SUM(CLM_PMT_AMT) FROM inpatient_claims
                                  GROUP BY 1)
        SELECT c.person, total FROM c""",
    "cte_select_star": """
        WITH b08 AS (SELECT * FROM beneficiary WHERE year = 2008)
        SELECT SP_STATE_CODE, COUNT(*) FROM b08 GROUP BY SP_STATE_CODE""",
    "subquery_alias": """
        SELECT s.state, s.n FROM (SELECT SP_STATE_CODE AS state, COUNT(*) AS n
                                  FROM beneficiary GROUP BY 1) AS s WHERE s.n > 5""",
    "window_function_and_qualify": """
        SELECT CLM_ID, ROW_NUMBER() OVER (PARTITION BY DESYNPUF_ID ORDER BY CLM_FROM_DT) AS rn,
               SUM(CLM_PMT_AMT) OVER (PARTITION BY DESYNPUF_ID) AS person_total
        FROM inpatient_claims QUALIFY rn = 1""",
    "lag_over_cte": """
        WITH y AS (SELECT year, SUM(MEDREIMB_IP) AS total FROM beneficiary GROUP BY year)
        SELECT year, total - LAG(total) OVER (ORDER BY year) AS change FROM y""",
    "list_contains_over_family": """
        SELECT COUNT(*) FROM inpatient_claims
        WHERE list_contains([ICD9_DGNS_CD_1, ICD9_DGNS_CD_2], '4019')""",
    "union_with_order_by_alias": """
        SELECT CLM_ID AS claim, CLM_PMT_AMT AS paid FROM inpatient_claims
        UNION ALL SELECT CLM_ID, CLM_PMT_AMT FROM outpatient_claims ORDER BY paid DESC""",
    "union_inside_cte": """
        WITH allc AS (SELECT DESYNPUF_ID, CLM_PMT_AMT FROM inpatient_claims
                      UNION ALL SELECT DESYNPUF_ID, CLM_PMT_AMT FROM outpatient_claims)
        SELECT DESYNPUF_ID, SUM(CLM_PMT_AMT) FROM allc GROUP BY DESYNPUF_ID""",
    "correlated_exists": """
        SELECT COUNT(*) FROM beneficiary b WHERE b.year = 2008 AND EXISTS (
          SELECT 1 FROM inpatient_claims i WHERE i.DESYNPUF_ID = b.DESYNPUF_ID AND year = 2008)""",
    "scalar_subqueries_without_from": """
        SELECT (SELECT COUNT(DISTINCT CLM_ID) FROM carrier_claims) * 1.0
             / (SELECT COUNT(*) FROM beneficiary WHERE year = 2008)""",
    "in_subquery": """
        SELECT COUNT(*) FROM carrier_claims WHERE DESYNPUF_ID IN (
          SELECT DESYNPUF_ID FROM beneficiary WHERE SP_DIABETES = 1)""",
    "star_and_qualified_star": "SELECT b.*, i.CLM_ID FROM beneficiary b, inpatient_claims i",
    "casts_and_date_arithmetic": """
        SELECT CAST(CLM_FROM_DT / 10000 AS INTEGER) AS yr,
               CAST(strptime(CAST(CLM_THRU_DT AS VARCHAR), '%Y%m%d') AS DATE) AS thru,
               COALESCE(CLM_UTLZTN_DAY_CNT, 0) FROM inpatient_claims""",
    "case_expression": """
        SELECT CASE WHEN BENE_SEX_IDENT_CD = '1' THEN 'M' ELSE 'F' END AS sex, COUNT(*)
        FROM beneficiary GROUP BY sex""",
    "table_function_and_unnest": """
        SELECT r.range, u.code FROM range(3) r,
               unnest(['a', 'b']) AS u(code), beneficiary WHERE year = 2008""",
    "lambda_parameter": """
        SELECT list_transform([LINE_NCH_PMT_AMT_1, LINE_NCH_PMT_AMT_2], x -> COALESCE(x, 0))
        FROM carrier_claims""",
    "recursive_cte": """
        WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 3)
        SELECT i FROM n""",
    "comments_and_semicolon": "-- count people\nSELECT COUNT(*) /* all */ FROM beneficiary;",
}

INVALID = {
    "fake_column": ("SELECT PATIENT_AGE FROM beneficiary", [], ["PATIENT_AGE"]),
    "fake_table": ("SELECT * FROM pharmacy_claims", ["pharmacy_claims"], []),
    "fake_table_columns_not_double_counted": (
        "SELECT p.DRUG_NAME, COST FROM prescriptions p WHERE p.COST > 5", ["prescriptions"], []),
    "column_on_wrong_table_qualified": (
        "SELECT b.CLM_PMT_AMT FROM beneficiary b", [], ["beneficiary.CLM_PMT_AMT"]),
    "column_on_wrong_table_in_join": (
        """SELECT i.year FROM beneficiary b JOIN inpatient_claims i
           ON i.DESYNPUF_ID = b.DESYNPUF_ID""", [], ["inpatient_claims.year"]),
    "unqualified_column_of_unreferenced_table": (
        "SELECT SUM(CLM_PMT_AMT) FROM beneficiary", [], ["CLM_PMT_AMT"]),
    "fake_column_in_where": (
        "SELECT COUNT(*) FROM beneficiary WHERE BENE_AGE > 65", [], ["BENE_AGE"]),
    "fake_column_in_group_and_order": (
        "SELECT COUNT(*) FROM inpatient_claims GROUP BY HOSPITAL_NAME ORDER BY ADMIT_DATE",
        [], ["HOSPITAL_NAME", "ADMIT_DATE"]),
    "family_member_that_does_not_exist": (
        "SELECT ICD9_DGNS_CD_3 FROM inpatient_claims", [], ["ICD9_DGNS_CD_3"]),
    "fake_column_inside_cte_body": (
        "WITH c AS (SELECT CLM_ID, TOTAL_COST AS paid FROM inpatient_claims) SELECT paid FROM c",
        [], ["TOTAL_COST"]),
    "column_not_exposed_by_cte_qualified": (
        "WITH c AS (SELECT CLM_ID FROM inpatient_claims) SELECT c.CLM_PMT_AMT FROM c",
        [], ["c.CLM_PMT_AMT"]),
    "column_not_exposed_by_cte_unqualified": (
        "WITH c AS (SELECT CLM_ID FROM inpatient_claims) SELECT CLM_PMT_AMT FROM c",
        [], ["CLM_PMT_AMT"]),
    "alias_from_inner_query_misspelt_outside": (
        "SELECT totl FROM (SELECT SUM(CLM_PMT_AMT) AS total FROM inpatient_claims) s",
        [], ["totl"]),
    "fake_table_in_subquery": (
        """SELECT COUNT(*) FROM beneficiary WHERE DESYNPUF_ID IN (
             SELECT DESYNPUF_ID FROM hospice_claims)""", ["hospice_claims"], []),
    "fake_column_in_scalar_subquery": (
        "SELECT (SELECT MAX(DISCHARGE_STATUS) FROM inpatient_claims)", [], ["DISCHARGE_STATUS"]),
    "fake_column_in_second_union_branch": (
        """SELECT CLM_ID FROM inpatient_claims
           UNION ALL SELECT CLAIM_NUMBER FROM outpatient_claims""", [], ["CLAIM_NUMBER"]),
    "fake_column_in_window": (
        """SELECT ROW_NUMBER() OVER (PARTITION BY MEMBER_ID ORDER BY CLM_FROM_DT)
           FROM carrier_claims""", [], ["MEMBER_ID"]),
    "double_quoted_text_is_an_identifier": (
        'SELECT COUNT(*) FROM beneficiary WHERE SP_STATE_CODE = "05"', [], ["05"]),
    "fake_table_and_fake_column_together": (
        """SELECT b.GENDER FROM beneficiary b JOIN enrollment e
           ON e.DESYNPUF_ID = b.DESYNPUF_ID""", ["enrollment"], ["beneficiary.GENDER"]),
    "same_unknown_reported_once": (
        "SELECT age, AGE FROM beneficiary WHERE Age > 1 ORDER BY age", [], ["age"]),
    "non_query_with_unknown_table": ("DROP TABLE claims_archive", ["claims_archive"], []),
}  # fmt: skip


@pytest.mark.parametrize("name", VALID)
def test_valid_sql_passes(name):
    result = check_sql(VALID[name], SCHEMA)
    assert result.passed, result.to_dict()


@pytest.mark.parametrize("name", INVALID)
def test_invalid_sql_reports_exactly_the_unknown_names(name):
    sql, tables, columns = INVALID[name]
    result = check_sql(sql, SCHEMA)
    assert not result.passed
    assert result.parse_error is None
    assert result.unknown_tables == tables
    assert result.unknown_columns == columns


@pytest.mark.parametrize("sql", ["SELEC x FROM beneficiary", "SELECT FROM WHERE", "", "   ;  "])
def test_unparseable_sql_gives_parse_error(sql):
    result = check_sql(sql, SCHEMA)
    assert not result.passed and result.parse_error
    assert result.unknown_tables == [] and result.unknown_columns == []


def test_enough_cases():
    assert len(VALID) + len(INVALID) >= 25


def test_known_table_in_a_write_statement_passes_v1():
    # V1 only judges names; refusing to run a DELETE is V2's job.
    assert check_sql("DELETE FROM beneficiary WHERE year = 2008", SCHEMA).passed


def test_result_to_dict():
    result = check_sql("SELECT nope FROM nowhere", SCHEMA)
    assert result.to_dict() == {
        "passed": False, "unknown_tables": ["nowhere"], "unknown_columns": [],
        "parse_error": None,
    }  # fmt: skip


def test_schema_from_catalogue_expands_families_and_keeps_excluded_columns():
    catalogue = load_catalogue(Path(__file__).parent / "fixtures" / "catalogue.yaml")
    schema = schema_from_catalogue(catalogue)
    assert schema["claim"] == {
        "ID", "CLM_ID", "DGNS_CD_1", "DGNS_CD_2", "DGNS_CD_3", "AMT", "EMPTY_CD",
    }  # fmt: skip
    assert "YEAR" in schema["person"]
    # EMPTY_CD has include: false, but it is a real column, so using it is not a hallucination.
    assert check_sql("SELECT EMPTY_CD, dgns_cd_2 FROM Claim", schema).passed
    assert check_sql("SELECT DGNS_CD_4 FROM claim", schema).unknown_columns == ["DGNS_CD_4"]
