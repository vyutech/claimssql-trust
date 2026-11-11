"""P3 validator tests. The questions are dummies about the made-up fixture data."""

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml

from claimssql.db import load_synpuf as ls
from claimssql.db.connection import get_connection
from claimssql.eval import validate_benchmark as vb
from claimssql.eval.compare import result_hash

FIXTURES = Path(__file__).parent / "fixtures"
TAGS = {"multi_year_beneficiary", "claim_segments"}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("p3") / "claims.duckdb"
    ls.build_database(FIXTURES / "raw", path)
    return path


@pytest.fixture()
def con(db_path):
    connection = get_connection(read_only=True, db_path=db_path)
    yield connection
    connection.close()


def q(**overrides) -> dict:
    """An answerable YAML record."""
    return {"id": "X-1", "group": "easy", "question": "dummy question", "sql": "SELECT 1",
            **overrides}  # fmt: skip


def inf(**overrides) -> dict:
    """An infeasible YAML record."""
    return {"id": "X-1", "group": "T1", "question": "dummy question", "notes": "dummy reason",
            **overrides}  # fmt: skip


def as_yaml(*records) -> str:
    return yaml.safe_dump(list(records), sort_keys=False)


def parse(*records):
    return vb.parse_yaml(as_yaml(*records), TAGS)


# ---------------------------------------------------------- YAML source ----


def test_fixture_file_parses_into_full_records():
    entries = vb.parse_yaml((FIXTURES / "questions.yaml").read_text(encoding="utf-8"), TAGS)
    assert [e.record.id for e in entries] == ["F-E1", "F-E2", "F-M1", "F-H1", "F-T1", "F-T4"]
    assert not any(e.errors or e.warnings for e in entries)

    e2 = entries[1].record
    assert e2.answerable is True and e2.trap_tags == ["multi_year_beneficiary"]
    assert e2.question == "fixture question 2: with a colon"
    # Multi-line block: inner line breaks and comments kept, final newline removed.
    assert e2.gold_sql == (
        "SELECT DESYNPUF_ID, year\nFROM beneficiary   -- comment kept\nORDER BY 1, 2"
    )
    assert e2.notes == 'keep "this" note'
    assert e2.gold_result_hash is None and e2.split is None

    t4 = entries[5].record
    assert (t4.answerable, t4.gold_sql, t4.trap_tags) == (False, None, [])
    assert t4.notes == "fixture reason\non two lines"
    assert entries[1].line == 8  # where "- id: F-E2" starts in the file


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (q(group="T1"), "infeasible, so it must have no sql"),
        (inf(group="hard"), "answerable but the record has no sql"),
        (q(sql="   "), "answerable but the record has no sql"),
        (q(group="extreme"), "group"),
        (q(tags=["made_up_tag"]), "unknown tags: made_up_tag"),
        (q(question="  "), "question is empty"),
        (q(id="bad id/../x"), "id"),
        (q(dificulty="easy"), "dificulty"),  # misspelt extra field
        (q(gold_sql="SELECT 1"), "gold_sql"),  # JSONL field names are not YAML fields
        (q(notes=2008), "notes"),  # must be text
        ({"id": "X-1"}, "question"),  # missing fields
        ("just a string", "not a record"),
    ],
)
def test_invalid_records_are_reported(bad, message):
    (entry,) = parse(bad)
    assert entry.record is None
    assert any(message in error for error in entry.errors), entry.errors


def test_file_level_problems_raise():
    with pytest.raises(vb.BenchmarkFileError, match="not valid YAML"):
        vb.parse_yaml("- id: X-1\n  question: a: b: c\n   bad indent", TAGS)
    with pytest.raises(vb.BenchmarkFileError, match="must be a list"):
        vb.parse_yaml("id: X-1\nquestion: not in a list", TAGS)
    assert vb.parse_yaml("", TAGS) == []
    assert vb.parse_yaml("# only a comment\n", TAGS) == []


def test_duplicate_ids_invalidate_every_copy():
    entries = parse(q(id="D-1"), q(id="D-2"), q(id="D-1", question="other"))
    assert [bool(e.record) for e in entries] == [False, True, False]
    assert "duplicate id 'D-1'" in entries[0].errors[0]


def test_infeasible_without_reason_is_a_warning():
    (entry,) = parse({"id": "X-1", "group": "T2", "question": "dummy"})
    assert entry.record is not None
    assert entry.warnings == ["infeasible record has no reason in notes"]


def test_real_trap_tags_file_loads():
    tags = vb.load_trap_tags()
    added = ["payment_type", "partial_enrollment", "date_basis", "death_attrition"]
    for tag in ["multi_year_beneficiary", "claim_segments", "wide_line_items", *added]:
        assert tag in tags
    assert len(tags) == 14 and all(meaning.strip() for meaning in tags.values())


# ------------------------------------------------------------ gold SQL ----


def test_gold_sql_writes_hash_and_preview(con, tmp_path):
    entries = parse(
        q(id="G-1", sql="SELECT count(*) AS n FROM beneficiary"),
        q(id="G-2", sql="SELECT DESYNPUF_ID, year FROM beneficiary ORDER BY 1, 2"),
        inf(id="G-3"),
    )
    vb.run_gold_sql(entries, con, timeout_s=10, preview_dir=tmp_path)

    assert entries[0].record.gold_result_hash == result_hash([(6,)])
    assert len(entries[1].record.gold_result_hash) == 64
    assert entries[2].record.gold_result_hash is None
    assert not any(e.warnings or e.gold_error for e in entries)

    with (tmp_path / "G-2.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["DESYNPUF_ID", "year"]
    assert rows[1] == ["000A", "2008"] and len(rows) == 1 + 6
    assert not (tmp_path / "G-3.csv").exists()


def test_preview_keeps_only_the_first_20_rows(con, tmp_path):
    entries = parse(q(id="P-1", sql="SELECT * FROM range(50)"))
    vb.run_gold_sql(entries, con, timeout_s=10, preview_dir=tmp_path)
    assert len((tmp_path / "P-1.csv").read_text().splitlines()) == 1 + 20


def test_gold_sql_failure_is_reported_and_gets_no_hash(con, tmp_path):
    entries = parse(
        q(id="B-1", sql="SELECT nope FROM beneficiary"),
        q(id="B-2", sql="DELETE FROM beneficiary"),  # the connection is read-only
    )
    entries[0].record = entries[0].record.model_copy(update={"gold_result_hash": "old"})
    vb.run_gold_sql(entries, con, timeout_s=10, preview_dir=tmp_path)
    assert "nope" in entries[0].gold_error
    assert entries[0].record.gold_result_hash is None  # no stale hash to score against
    assert entries[1].gold_error is not None
    assert list(tmp_path.iterdir()) == []
    assert con.execute("SELECT count(*) FROM beneficiary").fetchone()[0] == 6


def test_suspicious_results_are_warned(con):
    entries = parse(
        q(id="W-1", sql="SELECT * FROM beneficiary WHERE year = 1999"),
        q(id="W-2", sql="SELECT max(BENE_DEATH_DT) FROM beneficiary WHERE year = 2009"),
        q(id="W-3", sql="SELECT * FROM range(30)"),
    )
    vb.run_gold_sql(entries, con, timeout_s=10, max_rows=10)
    assert entries[0].warnings == ["gold result is empty (0 rows)"]
    assert entries[1].warnings == ["gold result is a single NULL"]
    assert "more than 10 rows" in entries[2].warnings[0]


def test_gold_sql_timeout(con):
    slow = "SELECT sum(a.range * b.range) FROM range(200000) a, range(200000) b"
    entries = parse(q(id="T-1", sql=slow))
    vb.run_gold_sql(entries, con, timeout_s=0.3)
    assert entries[0].gold_error.startswith("timeout")


# ------------------------------------------- memory in questions.jsonl ----


def jsonl(**fields) -> str:
    base = {"id": "X-1", "question": "dummy question", "group": "easy", "answerable": True,
            "gold_sql": "SELECT 1", "gold_result_hash": None, "trap_tags": [], "notes": "",
            "split": None}  # fmt: skip
    return json.dumps({**base, **fields})


def test_split_is_carried_over_from_the_previous_jsonl(con):
    previous = vb.load_previous(
        jsonl(id="K-1", split="dev", gold_result_hash="stale") + "\n"
        + jsonl(id="K-2", split="test", gold_sql="SELECT 2", gold_result_hash="old") + "\n"
        + "this line is broken\n"
    )
    entries = parse(q(id="K-1"), q(id="K-2", sql="SELECT 3"), q(id="K-3"))
    vb.apply_previous(entries, previous)
    vb.run_gold_sql(entries, con, timeout_s=10)
    vb.assign_splits(entries, seed=42, dev_fraction=0.2)

    k1, k2, k3 = (e.record for e in entries)
    assert (k1.split, k2.split) == ("dev", "test")  # kept, even though K-2's SQL changed
    assert k3.split in {"dev", "test"}
    assert k1.gold_result_hash == result_hash([(1,)])
    # Same SQL, different result -> warning. Changed SQL -> a new hash is expected.
    assert entries[0].warnings == ["gold result changed since the last run (same SQL)"]
    assert entries[1].warnings == []


def test_invalid_record_keeps_its_previous_line_and_removed_ids_are_noted():
    old_line = jsonl(id="K-1", split="dev", notes="old text")
    previous = vb.load_previous(old_line + "\n" + jsonl(id="GONE", split="test") + "\n")
    entries = parse(q(id="K-1", group="T1"), q(id="NEW-1"))  # K-1 is now invalid
    vb.apply_previous(entries, previous)
    text, notes = vb.render_jsonl(entries, previous)

    lines = text.splitlines()
    assert lines[0] == old_line  # split not lost because of a typo
    assert json.loads(lines[1])["id"] == "NEW-1"
    assert any("K-1: invalid now; previous version kept" in n for n in notes)
    assert any("GONE: no longer in questions.yaml" in n and "'test'" in n for n in notes)


def test_render_jsonl_record_format_and_order():
    entries = parse(q(id="R-2", sql="SELECT  1\t-- spacing kept", notes='quote " and é'),
                    {"id": 5}, inf(id="R-1"))  # fmt: skip
    text, notes = vb.render_jsonl(entries, {})
    first, last = (json.loads(line) for line in text.splitlines())  # invalid record: no line
    assert [first["id"], last["id"]] == ["R-2", "R-1"]
    assert first["gold_sql"] == "SELECT  1\t-- spacing kept"
    assert first["notes"] == 'quote " and é' and "é" in text
    assert last["answerable"] is False and last["gold_sql"] is None
    assert list(first) == [
        "id", "question", "group", "answerable", "gold_sql", "gold_result_hash",
        "trap_tags", "notes", "split",
    ]  # fmt: skip
    assert notes == [] and vb.render_jsonl([], {}) == ("", [])


# --------------------------------------------------------------- split ----


def many(group: str, n: int) -> list[dict]:
    make = q if group in vb.ANSWERABLE_GROUPS else inf
    return [make(id=f"{group}-{i:03d}", group=group) for i in range(n)]


def test_split_is_stratified_20_percent_per_group():
    entries = parse(*many("easy", 40), *many("T1", 25), *many("T2", 15))
    assert vb.assign_splits(entries, seed=42, dev_fraction=0.2) == 80
    for group, expected_dev in (("easy", 8), ("T1", 5), ("T2", 3)):
        splits = [e.record.split for e in entries if e.record.group == group]
        assert splits.count("dev") == expected_dev
        assert splits.count("test") == len(splits) - expected_dev


def test_split_is_deterministic_and_seeded():
    def dev_ids(seed, reverse=False):
        records = many("easy", 40)
        entries = parse(*(records[::-1] if reverse else records))  # file order must not matter
        vb.assign_splits(entries, seed=seed, dev_fraction=0.2)
        return {e.record.id for e in entries if e.record.split == "dev"}

    assert dev_ids(42) == dev_ids(42) == dev_ids(42, reverse=True)
    assert dev_ids(42) != dev_ids(7)


def test_existing_splits_are_never_changed():
    entries = parse(*many("easy", 20))
    for entry in entries[:10]:  # already more dev than 20% of 20
        entry.record = entry.record.model_copy(update={"split": "dev"})
    assert vb.assign_splits(entries, seed=42, dev_fraction=0.2) == 10
    assert all(e.record.split == "dev" for e in entries[:10])
    assert all(e.record.split == "test" for e in entries[10:])  # no extra dev needed
    assert vb.assign_splits(entries, seed=42, dev_fraction=0.2) == 0  # nothing left to assign


def test_composition_rows():
    entries = parse(*many("easy", 3), *many("T4", 2), q(id="BAD", group="T1"))
    for entry in entries[:3]:
        entry.record = entry.record.model_copy(update={"split": "dev"})
    assert vb.composition(entries, {"easy": 40, "T4": 15}) == [
        ("easy", 40, 3, -37, 3, 0, 0),
        ("T4", 15, 2, -13, 0, 0, 2),
        ("total", 55, 5, -50, 3, 0, 2),
    ]


# ----------------------------------------------------------------- CLI ----


def run_cli(tmp_path, db_path, *extra) -> tuple[int, Path, Path]:
    source = tmp_path / "questions.yaml"
    if not source.exists():
        shutil.copy(FIXTURES / "questions.yaml", source)
    output = tmp_path / "questions.jsonl"
    code = vb.main([
        "--questions", str(source), "--output", str(output), "--db-path", str(db_path),
        "--previews-dir", str(tmp_path / "previews"), "--out-dir", str(tmp_path / "tables"),
        *extra,
    ])  # fmt: skip
    return code, source, output


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_end_to_end(tmp_path, db_path, capsys):
    fixture_bytes = (FIXTURES / "questions.yaml").read_bytes()
    code, source, output = run_cli(tmp_path, db_path)
    assert code == 0
    assert source.read_bytes() == fixture_bytes  # the owner's file is never modified

    records = read_jsonl(output)
    assert [r["id"] for r in records] == ["F-E1", "F-E2", "F-M1", "F-H1", "F-T1", "F-T4"]
    assert all(r["gold_result_hash"] for r in records if r["answerable"])
    assert all(r["gold_result_hash"] is None for r in records if not r["answerable"])
    assert all(r["split"] in {"dev", "test"} for r in records)
    assert sorted(p.name for p in (tmp_path / "previews").iterdir()) == [
        "F-E1.csv", "F-E2.csv", "F-H1.csv", "F-M1.csv",
    ]  # fmt: skip
    with (tmp_path / "tables" / "benchmark_composition.csv").open(encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    assert table[0]["group"] == "easy" and table[0]["count"] == "2"
    assert table[-1]["group"] == "total" and table[-1]["target"] == "200"
    assert "Gold SQL failures: 0" in capsys.readouterr().out

    # Running again changes nothing; editing a question keeps its split.
    snapshot = output.read_text(encoding="utf-8")
    assert run_cli(tmp_path, db_path)[0] == 0
    assert output.read_text(encoding="utf-8") == snapshot

    source.write_text(
        source.read_text(encoding="utf-8").replace("fixture question 1", "reworded question"),
        encoding="utf-8",
    )
    assert run_cli(tmp_path, db_path)[0] == 0
    after = read_jsonl(output)
    assert after[0]["question"] == "reworded question"
    assert [r["split"] for r in after] == [r["split"] for r in records]


def test_cli_no_split_and_dry_run(tmp_path, db_path):
    code, source, output = run_cli(tmp_path, db_path, "--dry-run")
    assert code == 0
    assert not output.exists()
    assert not (tmp_path / "previews").exists() and not (tmp_path / "tables").exists()

    code, source, output = run_cli(tmp_path, db_path, "--no-split")
    records = read_jsonl(output)
    assert [r["split"] for r in records] == [None] * 6
    assert records[0]["gold_result_hash"] is not None


def test_cli_problems_give_exit_code_1(tmp_path, db_path, capsys):
    source = tmp_path / "questions.yaml"
    source.write_text("", encoding="utf-8")
    code, _, output = run_cli(tmp_path, db_path)
    assert code == 0 and output.read_text(encoding="utf-8") == ""  # empty benchmark is fine

    source.write_text(as_yaml(q(id="C-1", sql="SELECT nope"), q(id="C-2", group="T9")),
                      encoding="utf-8")  # fmt: skip
    assert run_cli(tmp_path, db_path)[0] == 1
    out = capsys.readouterr().out
    assert "Invalid records: 1" in out and "Gold SQL failures: 1" in out
    assert "C-2 (questions.yaml line" in out
    (written,) = read_jsonl(output)  # failing SQL: record written without a hash
    assert written["id"] == "C-1" and written["gold_result_hash"] is None

    source.write_text("- id: [unclosed", encoding="utf-8")
    before = output.read_text(encoding="utf-8")
    assert run_cli(tmp_path, db_path)[0] == 1
    assert "not valid YAML" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == before  # nothing written
