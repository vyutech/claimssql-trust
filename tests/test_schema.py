"""P2 tests: codebook parser, catalogue loader/validator, skeleton builder, context.

Everything runs on made-up fixtures: a small catalogue, a two-table in-memory
database, short codebook-style text, and the tiny P1 fixture CSVs.
"""

from pathlib import Path

import duckdb
import pytest
from pydantic import ValidationError

from claimssql.db import load_synpuf as ls
from claimssql.db import profile as pf
from claimssql.db.connection import get_connection
from claimssql.schema import codebook as cb
from claimssql.schema import context_builder as ctx
from claimssql.schema import skeleton as sk
from claimssql.schema.catalogue import (
    ColumnEntry,
    TableEntry,
    load_catalogue,
    validate_against_db,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def catalogue():
    return load_catalogue(FIXTURES / "catalogue.yaml")


@pytest.fixture()
def con():
    """The database the fixture catalogue describes."""
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE person (ID VARCHAR, SEX_CD VARCHAR, BIRTH_DT BIGINT, year INTEGER)"
    )
    connection.execute(
        "INSERT INTO person VALUES ('A', '1', 19400101, 2008), ('B', '2', NULL, 2008)"
    )
    connection.execute(
        "CREATE TABLE claim (ID VARCHAR, CLM_ID VARCHAR, DGNS_CD_1 VARCHAR, DGNS_CD_2 VARCHAR, "
        "DGNS_CD_3 VARCHAR, AMT DOUBLE, EMPTY_CD VARCHAR)"
    )
    connection.execute("INSERT INTO claim VALUES ('A', '001', '4019', NULL, NULL, 10.5, NULL)")
    yield connection
    connection.close()


# ------------------------------------------------------------- loader ----


def test_load_catalogue(catalogue):
    assert [t.name for t in catalogue.tables] == ["person", "claim"]
    assert catalogue.n_entries() == 9
    sex = catalogue.table("person").columns[1]
    assert sex.codes == {"1": "Male", "2": "Female"}  # integer YAML keys became text
    assert catalogue.table("person").columns[3].description == ""  # empty YAML value


def test_family_expands_to_real_columns(catalogue):
    family = catalogue.table("claim").columns[2]
    assert family.is_family
    assert family.members() == ["DGNS_CD_1", "DGNS_CD_2", "DGNS_CD_3"]
    assert ColumnEntry(name="CLM_ID", type="VARCHAR").members() == ["CLM_ID"]
    types = catalogue.table("claim").column_types(included_only=True)
    assert "DGNS_CD_2" in types and "EMPTY_CD" not in types


def test_bad_entries_are_rejected():
    with pytest.raises(ValidationError):
        ColumnEntry(name="DGNS_CD_5..2", type="VARCHAR")
    with pytest.raises(ValidationError):
        TableEntry(name="t", columns=[])


# ---------------------------------------------------------- validation ----


def test_matching_catalogue_has_only_warnings(catalogue, con):
    report = validate_against_db(catalogue, con)
    assert not report.has_mismatches
    assert report.empty_descriptions == [("person", "year")]
    assert report.values_without_code == [] and report.codes_not_in_data == []


def test_each_kind_of_mismatch_is_reported(catalogue, con):
    person, claim = catalogue.table("person"), catalogue.table("claim")
    person.columns.pop(2)  # BIRTH_DT no longer described
    person.columns.append(ColumnEntry(name="GHOST", type="VARCHAR", description="x"))
    claim.columns[3].type = "BIGINT"  # AMT is DOUBLE in the DB
    claim.columns.append(ColumnEntry(name="DGNS_CD_2", type="VARCHAR", description="again"))
    catalogue.tables.append(
        TableEntry(name="nowhere", grain="g", description="d",
                   columns=[ColumnEntry(name="X", type="VARCHAR", description="x")])
    )
    con.execute("CREATE TABLE extra (X INTEGER)")

    report = validate_against_db(catalogue, con)
    assert report.has_mismatches
    assert report.columns_not_in_catalogue == [("person", "BIRTH_DT")]
    assert report.columns_not_in_db == [("person", "GHOST")]
    assert report.type_mismatches == [("claim", "AMT", "BIGINT", "DOUBLE")]
    assert report.duplicate_columns == [("claim", "DGNS_CD_2")]
    assert report.tables_not_in_db == ["nowhere"]
    assert report.tables_not_in_catalogue == ["extra"]
    assert any(line.startswith("MISMATCH") for line in report.lines())


def test_codes_are_compared_with_the_data(catalogue, con):
    con.execute("INSERT INTO person VALUES ('C', '9', NULL, 2009)")
    catalogue.table("person").columns[1].codes["7"] = "Never used"
    report = validate_against_db(catalogue, con)
    assert not report.has_mismatches  # code gaps are warnings
    assert report.values_without_code == [("person", "SEX_CD", ["9"])]
    assert report.codes_not_in_data == [("person", "SEX_CD", ["7"])]


def test_column_names_match_case_insensitively(catalogue, con):
    catalogue.table("person").columns[3].name = "YEAR"
    assert not validate_against_db(catalogue, con).has_mismatches


# ------------------------------------------------------------- context ----


def test_render_context(catalogue):
    text = ctx.render_context(catalogue)
    assert "TABLE person\ngrain: one row per person per year\nMade-up people.\ncolumns:" in text
    assert "  SEX_CD VARCHAR: Sex [1=Male, 2=Female]" in text
    assert "  DGNS_CD_1 .. DGNS_CD_3 (3 columns) VARCHAR: Diagnosis Code" in text
    assert "  year INTEGER\n" in text  # no description -> no trailing colon
    assert "EMPTY_CD" not in text  # include: false
    assert "SECRET TRAP NOTE" not in text  # notes never reach the prompt


def test_approx_tokens():
    assert ctx.approx_tokens("") == 0
    assert ctx.approx_tokens("abc") == 1
    assert ctx.approx_tokens("abcd") == 2  # rounds up


# ------------------------------------------------------------ codebook ----

CODEBOOK_TEXT = """\
Table of Contents
BEN-4.
 ................ 13
CMS Linkable 2008–2010 Medicare DE-SynPUF
CODEBOOK  |  13
BEN-4.
Label: DESYNPUF: Sex
Variable Name: BENE_SEX_IDENT_CD
Type: Char
Categories
Code Label
2008
Freq.
1 Male 1,033,995  44.44 1,018,225  44.44 1,002,049  44.43
2 Female 1,292,861  55.56 1,273,095  55.56 1,253,049  55.57
Total  2,326,856  100 2,291,320  100 2,255,098  100
Summary Statistics
■ Label: THE SEX OF THE BENEFICIARY
BEN-9.
Label: DESYNPUF: Total number of months of part A coverage for the beneficiary
Variable Name: BENE_HI_CVRAGE_TOT_MONS
Type: Num
Categories
0 0 106,015  4.56 144,617  6.31 126,518  5.61
12 12 2,087,325  89.71 2,138,927  93.35 2,065,349  91.59
Total  2,326,856  100 2,291,320  100 2,255,098  100
IP-4.
Label: DESYNPUF: Claims start date1
Variable Name: CLM_FROM_DT2

Codebook from ResDAC website:
IP-17.
Label: DESYNPUF: NCH Beneficiary Blood Deductible Liability Amount
Variable Name: NCH_BENE_BLOOD_DDCTBL_LBLTY_AM(T)41
Source
IP-21-30.
Label prefix: DESYNPUF: Claim Diagnosis Code
Variable prefix: ICD9_DGNS_CD
Label: DESYNPUF: Claim Diagnosis Code 1 – DESYNPUF: Claim Diagnosis Code 10
Variable Name: ICD9_DGNS_CD_1 – ICD9_DGNS_CD_10
Source
CAR-78-90.
Label prefix: DESYNPUF: Line Beneficiary Primary Payer Paid Amount
Variable prefix: LINE_BENE_PRMRY_PYR_PD_AMT53
Label: DESYNPUF: Line Beneficiary Primary Payer Paid Amount 1 – DESYNPUF: Line Beneficiary
Primary Payer Paid Amount 13
Variable Name: LINE_BENE_PRMRY_PYR_PD_AMT_1 –
LINE_BENE_PRMRY_PYR_PD_AMT_13
Source
CAR-117-129.
Label prefix: DESYNPUF: Line Processing Indicator Code
Label: DESYNPUF: Line Processing Indicator Code 1 – DESYNPUF: Line Processing Indicator Code 13
Variable Name: LINE_PRCSG_IND_CD_1 – LINE_PRCSG_IND_CD_13
Source
A  =  Allowed
B  =  Benefits exhausted
!  =  MSP cost avoided – COB Contractor ('00' 2–byte code)
A  =  A later table repeating the code
PDE-2.
Label: DESYNPUF: CCW Part D Event Number
Variable Name: PDE_ID
"""


@pytest.fixture(scope="module")
def entries():
    return {e.entry_id: e for e in cb.parse_codebook(CODEBOOK_TEXT)}


def test_codebook_single_variable_with_code_table(entries):
    sex = entries["BEN-4"]  # the contents-page "BEN-4." without a label is skipped
    assert (sex.table, sex.columns, sex.label) == ("beneficiary", ("BENE_SEX_IDENT_CD",), "Sex")
    assert sex.codes == {"1": "Male", "2": "Female"}


def test_codebook_footnote_digits_are_removed(entries):
    assert entries["IP-4"].columns == ("CLM_FROM_DT",)
    assert entries["IP-4"].label == "Claims start date"
    assert entries["IP-17"].columns == ("NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",)


def test_codebook_families(entries):
    dgns = entries["IP-21-30"]
    assert dgns.label == "Claim Diagnosis Code"
    assert dgns.columns[0] == "ICD9_DGNS_CD_1" and dgns.columns[-1] == "ICD9_DGNS_CD_10"
    wrapped = entries["CAR-78-90"]  # label and variable name both wrap onto two lines
    assert len(wrapped.columns) == 13
    assert wrapped.columns[-1] == "LINE_BENE_PRMRY_PYR_PD_AMT_13"
    assert wrapped.label == "Line Beneficiary Primary Payer Paid Amount"


def test_codebook_equals_style_codes_first_definition_wins(entries):
    codes = entries["CAR-117-129"].codes
    assert codes["A"] == "Allowed" and codes["B"] == "Benefits exhausted"
    assert codes["!"].startswith("MSP cost avoided")


def test_codebook_index_skips_out_of_scope_sections(entries):
    index = cb.index_by_column(list(entries.values()))
    assert index[("inpatient_claims", "ICD9_DGNS_CD_7")].entry_id == "IP-21-30"
    assert not any(column == "PDE_ID" for _, column in index)


# ------------------------------------------------------------ skeleton ----


def test_group_columns_splits_on_include_and_type():
    def col(name, typ="VARCHAR", include=True):
        return sk.ColumnInfo(name, typ, include, distinct_count=5)

    groups = sk.group_columns(
        [
            col("CLM_ID"),
            col("PRCDR_CD_1"), col("PRCDR_CD_2"), col("PRCDR_CD_3", include=False),
            col("AMT_1", "DOUBLE"), col("AMT_2", "BIGINT"),
            col("HCPCS_CD_1"), col("HCPCS_CD_3"),  # gap in the numbering
        ]
    )  # fmt: skip
    assert [sk.entry_name(g) for g in groups] == [
        "CLM_ID", "PRCDR_CD_1..2", "PRCDR_CD_3", "AMT_1", "AMT_2", "HCPCS_CD_1", "HCPCS_CD_3",
    ]  # fmt: skip


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """(connection, profile, codebook index) from the tiny P1 fixture CSVs."""
    db_path = tmp_path_factory.mktemp("p2") / "claims.duckdb"
    ls.build_database(FIXTURES / "raw", db_path)
    connection = get_connection(read_only=True, db_path=db_path)
    profile = {
        (p.table, p.column): (p.null_pct, p.distinct_count)
        for table in ls.existing_tables(connection)
        for p in pf.profile_table(connection, table)
    }
    index = cb.index_by_column(cb.parse_codebook(CODEBOOK_TEXT))
    yield connection, profile, index
    connection.close()


@pytest.fixture(scope="module")
def built(inputs):
    """Catalogue drafted from those inputs."""
    connection, profile, index = inputs
    catalogue, report = sk.build_catalogue(connection, profile, index)
    return catalogue, report, connection


def test_build_catalogue_fields(built):
    catalogue, report, connection = built
    bene = {e.name: e for e in catalogue.table("beneficiary").columns}
    assert bene["BENE_SEX_IDENT_CD"].description == "Sex"
    assert bene["BENE_SEX_IDENT_CD"].codes == {"1": "Male", "2": "Female"}
    assert bene["BENE_SEX_IDENT_CD"].type == "VARCHAR"
    assert all(e.notes == "" for t in catalogue.tables for e in t.columns)
    assert catalogue.table("beneficiary").grain == "one row per beneficiary per year"

    # Columns the (test) codebook does not cover are listed, description left empty.
    assert bene["year"].description == ""
    assert ("beneficiary", "year") in report.undescribed

    # A drafted catalogue always matches the database it was drafted from.
    assert not validate_against_db(catalogue, connection).has_mismatches


def test_build_catalogue_excludes_all_null_columns(built):
    catalogue, report, _ = built
    inpatient = {e.name: e for e in catalogue.table("inpatient_claims").columns}
    assert inpatient["HCPCS_CD_1"].include is False  # empty on every fixture row
    assert ("inpatient_claims", "HCPCS_CD_1", "100% NULL") in report.excluded
    assert inpatient["CLM_ID"].include is True


def test_build_catalogue_codes_only_for_values_in_data(built):
    catalogue, report, _ = built
    carrier = {e.name: e for e in catalogue.table("carrier_claims").columns}
    # Data has A, S, O. The codebook text explains A (and B, !, which are unused).
    assert carrier["LINE_PRCSG_IND_CD_1"].codes == {"A": "Allowed"}
    assert ("carrier_claims", "LINE_PRCSG_IND_CD_1", ["O", "S"]) in report.values_without_code
    # "0 -> 0, 12 -> 12" is not a code table.
    assert sk._is_identity({"00": "0", "12": "12"}) and not sk._is_identity({"1": "Yes"})


def test_distinct_guard_and_missing_profile(inputs):
    connection, profile, index = inputs
    # With a guard of 2, LINE_PRCSG_IND_CD_1 (3 distinct values) gets no codes,
    # while BENE_SEX_IDENT_CD (2 distinct values) keeps them.
    catalogue, _ = sk.build_catalogue(connection, profile, index, max_coded_distinct=2)
    carrier = {e.name: e for e in catalogue.table("carrier_claims").columns}
    bene = {e.name: e for e in catalogue.table("beneficiary").columns}
    assert carrier["LINE_PRCSG_IND_CD_1"].codes == {}
    assert bene["BENE_SEX_IDENT_CD"].codes == {"1": "Male", "2": "Female"}

    with pytest.raises(KeyError, match="re-run profile"):
        sk.build_catalogue(connection, {}, index)


def test_forced_exclude_applies(inputs, monkeypatch):
    connection, profile, index = inputs
    monkeypatch.setitem(sk.FORCED_EXCLUDES, ("inpatient_claims", "ICD9_PRCDR_CD_1"), "test reason")
    catalogue, report = sk.build_catalogue(connection, profile, index)
    entry = {e.name: e for e in catalogue.table("inpatient_claims").columns}["ICD9_PRCDR_CD_1"]
    assert entry.include is False
    assert ("inpatient_claims", "ICD9_PRCDR_CD_1", "test reason") in report.excluded
    assert ("inpatient_claims", "ICD9_PRCDR_CD_6") in sk.FORCED_EXCLUDES  # the real rule


def test_write_catalogue_round_trips_and_never_overwrites(built, tmp_path):
    catalogue, _, _ = built
    path = tmp_path / "catalogue.yaml"
    sk.write_catalogue(catalogue, path)
    assert path.read_text(encoding="utf-8").startswith("# Schema catalogue")
    assert load_catalogue(path) == catalogue

    path.write_text("owner edits", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sk.write_catalogue(catalogue, path)
    assert path.read_text(encoding="utf-8") == "owner edits"
