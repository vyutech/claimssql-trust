"""P6 scoring, metrics, taxonomy and report tests.

They run on a hand-made run (15 records for one model) small enough that every
expected number below can be checked by hand. The table in RUN shows, per
question, what the forced and the abstain prompt produced:

    ok      SQL whose result equals gold          ref   the model refused
    bad     SQL that runs, wrong result           pe    unreadable reply
    schema  SQL with an unknown column (V1 fails, V2 fails)
    exec    SQL with known names that fails to run (V1 passes, V2 fails)
"""

import csv
import json

import pytest

from claimssql.eval import metrics, report, taxonomy
from claimssql.eval.score import REGIONS, score_record, score_records

VALUE_SQL = "SELECT count(*) FROM beneficiary WHERE year = 2011"  # 50 characters
LOW_CARD = taxonomy.normalise_low_cardinality({"beneficiary": {"year": [2008, 2009, 2010]}})

#   id    group     split  forced  abstain   agreement  confidence  sql length
RUN = [
    ("t1", "easy", "test", "ok", "ok", 1.0, 90, 10),
    ("t2", "easy", "test", "ok", "ok", 0.4, 60, 20),
    ("t3", "medium", "test", "bad", "bad", 1.0, 90, 50),  # abstain SQL is VALUE_SQL
    ("t4", "medium", "test", "bad", "schema", 0.2, 80, 40),
    ("t5", "hard", "test", "bad", "exec", 0.2, 50, 50),
    ("t6", "hard", "test", "ok", "ref", 0.0, 0, 0),
    ("t7", "T1", "test", "bad", "ref", 0.0, 0, 0),
    ("t8", "T1", "test", "bad", "bad", 0.2, 70, 5),
    ("t9", "T2", "test", "bad", "bad", 1.0, 100, 25),
    ("t10", "T4", "test", "ref", "pe", 0.0, 0, 0),
    ("d1", "easy", "dev", "ok", "ok", 0.6, 90, 10),
    ("d2", "easy", "dev", "ok", "bad", 0.4, 90, 10),
    ("d3", "T1", "dev", "ref", "bad", 0.2, 90, 10),
    ("d4", "medium", "dev", "ok", "ok", 1.0, 90, 10),
    ("d5", "T3", "dev", "ref", "ref", 0.0, 0, 0),
]


def generation(kind: str, sql: str, confidence: int, latency: float) -> dict:
    attempt = {"raw": "", "latency_ms": latency, "prompt_tokens": 50, "completion_tokens": 10,
               "cached": False, "parse_problem": None}  # fmt: skip
    base = {"attempts": [attempt], "latency_ms": latency, "prompt_tokens": 50,
            "completion_tokens": 10, "error": None}  # fmt: skip
    if kind == "pe":
        return {**base, "parsed": None, "error": "PARSE_ERROR"}
    if kind == "ref":
        parsed = {"answerable": False, "sql": None, "confidence": confidence, "reason": "no"}
    else:
        parsed = {"answerable": True, "sql": sql, "confidence": confidence, "reason": ""}
    return {**base, "parsed": parsed}


def verifiers(kind: str, gold: str | None) -> tuple[dict | None, dict | None]:
    if kind in ("ref", "pe"):
        return None, None
    v1 = {"passed": kind != "schema", "unknown_tables": [], "parse_error": None,
          "unknown_columns": ["AGE"] if kind == "schema" else []}  # fmt: skip
    if kind in ("schema", "exec"):
        return v1, {"ok": False, "error_class": "binder", "error": "boom", "result_hash": None}
    answer = gold if kind == "ok" else "WRONG"
    return v1, {"ok": True, "error_class": None, "error": None, "result_hash": answer}


def make_record(qid, group, split, forced, abstain, agreement, confidence, length) -> dict:
    answerable = group in ("easy", "medium", "hard")
    gold = f"GOLD-{qid}" if answerable else None
    sql = VALUE_SQL if qid == "t3" else "S" * length
    forced_v1, forced_v2 = verifiers(forced, gold)
    v1, v2 = verifiers(abstain, gold)
    sample = generation("ok", sql, confidence, 100.0)
    return {
        "run_id": "fixture", "question_id": qid, "question": f"question {qid}",
        "group": group, "answerable": answerable, "split": split, "gold_result_hash": gold,
        "model_key": "m", "provider": "fake", "model": "fake-1", "model_options": {},
        "prompt_version": "v1",
        "forced": generation(forced, "SELECT 1", 100, 200.0),  # the forced call is slower
        "abstain": generation(abstain, sql, confidence, 100.0),
        "samples": [sample] * 5,
        "forced_v1": forced_v1, "forced_v2": forced_v2, "v1": v1, "v2": v2,
        "v3": {"agreement": agreement, "greedy_in_majority": agreement > 0},
    }  # fmt: skip


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return [make_record(*row) for row in RUN]


@pytest.fixture(scope="module")
def test_records(records) -> list[dict]:
    return [r for r in records if r["split"] == "test"]


def counts(cases) -> tuple[int, ...]:
    by_region = metrics.region_counts(cases)
    return tuple(by_region[region] for region in REGIONS)


# ----------------------------------------------------------------- score ----


def test_outcome_regions_per_config(test_records):
    # (answerable-correct, answerable-wrong, answerable-abstained,
    #  infeasible-abstained, infeasible-answered) over t1..t10
    expected = {
        "C0": (3, 3, 0, 1, 3),  # t1 t2 t6 right; t3 t4 t5 wrong; t7 t8 t9 answered; t10 refused
        "C1": (2, 3, 1, 2, 2),  # t6 and t7 now refuse
        "C2": (2, 2, 2, 2, 2),  # t4 stopped by V1
        "C3": (2, 1, 3, 2, 2),  # t5 stopped by V2
        "C4": (1, 1, 4, 3, 1),  # tau 0.6 also stops t2 (0.4) and t8 (0.2)
    }
    for config, want in expected.items():
        assert counts(score_records(test_records, config, 0.6)) == want, config


def test_score_record_details(test_records):
    t4 = next(r for r in test_records if r["question_id"] == "t4")
    c1 = score_record(t4, "C1")
    assert (c1.answered, c1.correct, c1.region) == (True, False, "answerable_wrong")
    c2 = score_record(t4, "C2")
    assert (c2.answered, c2.reason_code, c2.sql) == (False, "SCHEMA_UNKNOWN", None)
    # An infeasible question can never be "correct", whatever the SQL returns.
    t9 = next(r for r in test_records if r["question_id"] == "t9")
    assert score_record(t9, "C3").region == "infeasible_answered"


# --------------------------------------------------------------- metrics ----


def test_summary_numbers_by_hand(test_records):
    def row(config):
        s = metrics.summarise(score_records(test_records, config, 0.6))
        return (s["n"], s["answered"], round(s["ex"], 4), s["coverage"], round(s["risk"], 4),
                s["rs_0"], s["rs_1"], s["rs_10"])  # fmt: skip

    # n, answered, EX (of 6 answerable), coverage, risk, RS_0, RS_1, RS_10
    # RS_c = 100 * (correct + infeasible-abstained - c * harmful) / 10
    assert row("C0") == (10, 9, 0.5, 0.9, 0.6667, 40.0, -20.0, -560.0)
    assert row("C1") == (10, 7, 0.3333, 0.7, 0.7143, 40.0, -10.0, -460.0)
    assert row("C2") == (10, 6, 0.3333, 0.6, 0.6667, 40.0, 0.0, -360.0)
    assert row("C3") == (10, 5, 0.3333, 0.5, 0.6, 40.0, 10.0, -260.0)
    assert row("C4") == (10, 3, 0.1667, 0.3, 0.6667, 40.0, 20.0, -160.0)


def test_summary_edge_cases():
    empty = metrics.summarise([])
    assert empty["n"] == 0 and empty["ex"] is None and empty["rs_10"] is None
    refused = make_record("x", "T1", "test", "ref", "ref", 0.0, 0, 0)
    summary = metrics.summarise(score_records([refused], "C1"))
    assert summary["risk"] is None  # nothing answered: risk is undefined, not 0
    assert (summary["coverage"], summary["ex"], summary["rs_10"]) == (0.0, None, 100.0)


def test_tau_is_chosen_on_dev_by_rs_10(records):
    dev = [r for r in records if r["split"] == "dev"]
    tau, rows = metrics.choose_tau(dev)
    # d1 right (agreement 0.6), d2 wrong (0.4), d3 infeasible answered (0.2),
    # d4 right (1.0), d5 refused. RS_10 = 100 * (right + inf.abstained - 10 * harmful) / 5
    assert [(row["tau"], row["rs_10"]) for row in rows] == [
        (0.0, -340.0), (0.2, -340.0), (0.4, -120.0), (0.6, 80.0), (0.8, 60.0), (1.0, 60.0),
    ]  # fmt: skip
    assert tau == 0.6
    assert metrics.choose_tau([])[0] == metrics.FALLBACK_TAU


def test_tau_tie_goes_to_the_lowest_tau():
    always_right = [make_record("a", "easy", "dev", "ok", "ok", 1.0, 90, 10)]
    assert metrics.choose_tau(always_right)[0] == 0.0


def test_risk_coverage_points(test_records):
    rows = {row["tau"]: (row["answered"], row["coverage"]) for row in
            metrics.risk_coverage(test_records)}  # fmt: skip
    # C3 answers t1 t2 t3 t8 t9 (agreements 1.0, 0.4, 1.0, 0.2, 1.0).
    assert rows == {0.0: (5, 0.5), 0.2: (5, 0.5), 0.4: (4, 0.4), 0.6: (3, 0.3),
                    0.8: (3, 0.3), 1.0: (3, 0.3)}  # fmt: skip


def test_signal_auroc_by_hand(test_records):
    table = {row["signal"]: row for row in metrics.signal_table(test_records)}
    # C1 answers t1 t2 (right) and t3 t4 t5 t8 t9 (wrong): 7 answers, 5 wrong.
    assert all((row["n"], row["n_wrong"]) == (7, 5) for row in table.values())
    # AUROC = share of (wrong, right) pairs where the wrong one has the higher
    # signal, ties counting half. 10 pairs each.
    #   1 - agreement:    wrong 0, .8, .8, .8, 0   right 0, .6     -> 7 / 10
    #   100 - confidence: wrong 10, 20, 50, 30, 0  right 10, 40    -> 4.5 / 10
    #   sql length:       wrong 50, 40, 50, 5, 25  right 10, 20    -> 8 / 10
    assert table["1 - agreement"]["auroc"] == pytest.approx(0.7)
    assert table["100 - confidence"]["auroc"] == pytest.approx(0.45)
    assert table["sql length"]["auroc"] == pytest.approx(0.8)


def test_auroc_undefined_with_one_class():
    assert metrics.auroc([True, True], [0.1, 0.9]) is None
    assert metrics.auroc([], []) is None
    assert metrics.auroc([False, True], [0.1, 0.9]) == 1.0
    assert metrics.auroc([False, True], [0.9, 0.1]) == 0.0


def test_latency_and_cost(test_records):
    stats = metrics.latency_cost(test_records)
    # Every question: 1 forced call of 200 ms + 6 calls of 100 ms, 50 + 10 tokens each.
    assert stats["n_questions"] == 10 and stats["calls_per_question"] == 7
    assert stats["latency_p50_ms"] == 100 and stats["latency_p95_ms"] == 200
    assert stats["c1_latency_per_question_ms"] == 100  # the abstain greedy call
    assert stats["c4_latency_per_question_ms"] == 600  # ... plus the 5 samples
    assert stats["prompt_tokens_per_question"] == 350
    assert stats["completion_tokens_per_question"] == 70
    assert metrics.latency_cost([]) == {"n_questions": 0}


# -------------------------------------------------------------- taxonomy ----

LOW = taxonomy.normalise_low_cardinality(
    {
        "beneficiary": {"year": [2008, 2009, 2010], "BENE_SEX_IDENT_CD": ["1", "2"],
                        "SP_STATE_CODE": ["01", "05", "39"], "MEDREIMB_IP": [0.0, 10.0]},
        "inpatient_claims": {"SEGMENT": [1, 2]},
        "outpatient_claims": {"SEGMENT": [1, 2]},
    },
    skip={("beneficiary", "MEDREIMB_IP")},
)  # fmt: skip


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT count(*) FROM beneficiary WHERE year = 2011", ["beneficiary.YEAR = 2011"]),
        ("SELECT count(*) FROM beneficiary WHERE year = 2008", []),
        ("SELECT count(*) FROM beneficiary WHERE BENE_SEX_IDENT_CD = 'M'",
         ["beneficiary.BENE_SEX_IDENT_CD = 'M'"]),
        ("SELECT count(*) FROM beneficiary WHERE BENE_SEX_IDENT_CD = '1'", []),
        ("SELECT count(*) FROM beneficiary WHERE BENE_SEX_IDENT_CD = 1", []),  # cast by DuckDB
        # Text against text is exact: '5' is not the stored '05' ...
        ("SELECT count(*) FROM beneficiary WHERE SP_STATE_CODE = '5'",
         ["beneficiary.SP_STATE_CODE = '5'"]),
        # ... but the number 5 equals '05' once DuckDB casts it.
        ("SELECT count(*) FROM beneficiary WHERE SP_STATE_CODE = 5", []),
        ("SELECT count(*) FROM beneficiary WHERE year IN (2008, 2012, 2013)",
         ["beneficiary.YEAR = 2012", "beneficiary.YEAR = 2013"]),
        ("SELECT count(*) FROM beneficiary b WHERE b.year <> 1999", ["beneficiary.YEAR = 1999"]),
        ("SELECT count(*) FROM beneficiary WHERE 2011 = year", ["beneficiary.YEAR = 2011"]),
        ("SELECT count(*) FROM inpatient_claims WHERE SEGMENT = 3",
         ["inpatient_claims.SEGMENT = 3"]),
        # Skipped amount column; a column that is not low-cardinality; a range filter.
        ("SELECT count(*) FROM beneficiary WHERE MEDREIMB_IP = 99", []),
        ("SELECT count(*) FROM beneficiary WHERE DESYNPUF_ID = 'nobody'", []),
        ("SELECT count(*) FROM beneficiary WHERE year > 2050", []),
        # Two tables own SEGMENT: not judged. A bare column next to a CTE: not judged.
        ("SELECT count(*) FROM inpatient_claims i, outpatient_claims o WHERE SEGMENT = 3", []),
        ("WITH c AS (SELECT 2011 AS year) SELECT * FROM c, beneficiary WHERE year = 2011", []),
        # The check also looks inside CTEs and subqueries.
        ("WITH c AS (SELECT * FROM beneficiary WHERE year = 2020) SELECT count(*) FROM c",
         ["beneficiary.YEAR = 2020"]),
        ("SELEC nonsense (", []),
    ],
)  # fmt: skip
def test_value_mismatches(sql, expected):
    assert taxonomy.value_mismatches(sql, LOW) == expected


def test_taxonomy_labels_per_config(test_records):
    def labels(config):
        out = dict.fromkeys(taxonomy.LABELS, 0)
        for record, case in zip(test_records, score_records(test_records, config, 0.6),
                                strict=True):  # fmt: skip
            if (label := taxonomy.label_case(record, case, LOW_CARD)) is not None:
                out[label] += 1
        return tuple(out.values())

    # (infeasible_answered, schema, execution_error, value, logic)
    assert labels("C0") == (3, 0, 0, 0, 3)  # forced SQL "SELECT 1" runs: logic errors
    assert labels("C1") == (2, 1, 1, 1, 0)  # t4 schema, t5 execution, t3 year = 2011
    assert labels("C2") == (2, 0, 1, 1, 0)
    assert labels("C3") == (2, 0, 0, 1, 0)
    assert labels("C4") == (1, 0, 0, 1, 0)


def test_label_order_and_non_harmful_cases(test_records):
    t1, t9 = test_records[0], test_records[8]
    assert taxonomy.label_case(t1, score_record(t1, "C1"), LOW_CARD) is None  # correct answer
    # Infeasible beats everything else, even if the SQL also has a schema error.
    broken = {**t9, "v1": {"passed": False, "unknown_columns": ["X"], "parse_error": None}}
    assert taxonomy.label_case(broken, score_record(broken, "C1"), LOW_CARD) == \
        taxonomy.INFEASIBLE_ANSWERED  # fmt: skip


# ---------------------------------------------------------------- report ----


@pytest.fixture()
def report_dir(records, tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "tables"
    runs.mkdir()
    (runs / "fixture.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    low = tmp_path / "low.json"
    low.write_text(json.dumps({"beneficiary": {"year": [2008, 2009, 2010]}}), encoding="utf-8")
    code = report.main(["--run-id", "fixture", "--runs-dir", str(runs), "--out-dir", str(out),
                        "--low-cardinality", str(low)])  # fmt: skip
    assert code == 0
    return out


def read(path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_report_writes_every_table(report_dir):
    assert sorted(p.name for p in report_dir.iterdir()) == [
        "abstain_reasons.csv", "e1_ex_by_model_difficulty.csv", "e2_rs_by_config_penalty.csv",
        "e3_generation_failures.csv", "e3_taxonomy.csv", "e4_risk_coverage.csv",
        "e4_signal_auroc.csv", "e5_latency_cost.csv", "failures.jsonl",
        "outcomes_confusion.csv", "tau_selection.csv",
    ]  # fmt: skip


def test_report_counts_abstain_reasons_and_failed_generations(report_dir):
    reasons = {(r["split"], r["config"]): r for r in read(report_dir / "abstain_reasons.csv")}
    # Test split, C3: t6 t7 refused (MODEL_REFUSED), t10 unreadable (PARSE_ERROR),
    # t4 unknown column (SCHEMA_UNKNOWN), t5 does not run (EXEC_ERROR).
    c3 = reasons[("test", "C3")]
    got = [c3[k] for k in ("n_abstained", "MODEL_REFUSED", "PARSE_ERROR", "SCHEMA_UNKNOWN",
                           "EXEC_ERROR", "DEGENERATE_OUTPUT")]  # fmt: skip
    assert got == ["5", "2", "1", "1", "1", "0"]
    assert reasons[("test", "C4")]["LOW_AGREEMENT"] == "2"  # t2 and t8

    (test_row,) = [r for r in read(report_dir / "e3_generation_failures.csv")
                   if r["split"] == "test"]  # fmt: skip
    assert (test_row["questions"], test_row["generations"], test_row["parse_error"],
            test_row["degenerate_output"]) == ("10", "70", "1", "0")  # fmt: skip


def test_degenerate_output_is_counted_per_model():
    looped = {"attempts": [{"latency_ms": 900.0, "prompt_tokens": 0, "completion_tokens": 0,
                            "raw": "", "cached": False, "parse_problem": "looped"}],
              "latency_ms": 900.0, "prompt_tokens": 0, "completion_tokens": 0,
              "parsed": None, "error": "DEGENERATE_OUTPUT"}  # fmt: skip
    one = make_record("t1", "easy", "test", "ok", "ok", 0.8, 90, 10)
    one["samples"] = [*one["samples"][:2], looped, *one["samples"][3:]]
    two = make_record("t2", "hard", "test", "ok", "ok", 1.0, 90, 10)
    two["abstain"] = looped
    clean = make_record("t3", "easy", "test", "ok", "ok", 1.0, 90, 10)

    assert metrics.generation_failures([one, two, clean]) == {
        "questions": 3, "generations": 21, "parse_error": 0, "degenerate_output": 2,
        "degenerate_greedy": 1, "questions_with_degenerate": 2,
    }  # fmt: skip
    case = score_record(two, "C1")
    assert (case.answered, case.reason_code, case.region) == (
        False, "DEGENERATE_OUTPUT", "answerable_abstained",
    )  # fmt: skip
    # A loop is an abstention, never a harmful answer, so the taxonomy leaves it out.
    assert taxonomy.label_case(two, case, LOW_CARD) is None
    # Its latency still counts towards the cost of the question.
    assert metrics.latency_cost([two])["c1_latency_per_question_ms"] == 900


def test_report_tau_and_main_table(report_dir):
    chosen = [r for r in read(report_dir / "tau_selection.csv") if r["chosen"] == "yes"]
    assert [(r["model"], r["tau"], r["dev_n"], r["rs_10"]) for r in chosen] == [
        ("m", "0.6", "5", "80.0")
    ]

    e2 = {(r["split"], r["config"]): r for r in read(report_dir / "e2_rs_by_config_penalty.csv")}
    c4 = e2[("test", "C4")]
    assert (c4["tau"], c4["n"], c4["coverage"], c4["risk"], c4["ex"]) == (
        "0.6", "10", "0.3", "0.6667", "0.1667",
    )  # fmt: skip
    assert (c4["rs_0"], c4["rs_1"], c4["rs_10"]) == ("40.0", "20.0", "-160.0")
    assert e2[("test", "C0")]["rs_10"] == "-560.0" and e2[("test", "C0")]["tau"] == ""
    assert e2[("all", "C1")]["n"] == "15"  # all = test + dev


def test_report_other_tables(report_dir):
    e1 = {(r["split"], r["difficulty"]): r for r in
          read(report_dir / "e1_ex_by_model_difficulty.csv")}  # fmt: skip
    easy = e1[("test", "easy")]  # t1 right everywhere; t2 right until C4 (agreement 0.4)
    assert (easy["n_answerable"], easy["ex_C0"], easy["ex_C3"], easy["ex_C4"]) == (
        "2", "1.0", "1.0", "0.5",
    )  # fmt: skip
    assert e1[("test", "hard")]["ex_C0"] == "0.5" and e1[("test", "hard")]["ex_C1"] == "0.0"
    assert e1[("test", "all")]["n_answerable"] == "6"

    confusion = {(r["split"], r["config"]): r for r in read(report_dir / "outcomes_confusion.csv")}
    c3 = confusion[("test", "C3")]
    assert [c3[region] for region in REGIONS] == ["2", "1", "3", "2", "2"]

    e3 = {(r["split"], r["config"]): r for r in read(report_dir / "e3_taxonomy.csv")}
    c1 = e3[("test", "C1")]
    assert (c1["n_harmful"], c1["schema"], c1["execution_error"], c1["value"], c1["logic"],
            c1["infeasible_answered"]) == ("5", "1", "1", "1", "0", "2")  # fmt: skip

    auroc = {(r["split"], r["signal"]): r["auroc"] for r in
             read(report_dir / "e4_signal_auroc.csv")}  # fmt: skip
    assert auroc[("test", "1 - agreement")] == "0.7"

    curve = [r for r in read(report_dir / "e4_risk_coverage.csv") if r["split"] == "test"]
    assert [r["answered"] for r in curve] == ["5", "5", "4", "3", "3", "3"]

    (cost,) = read(report_dir / "e5_latency_cost.csv")
    assert (cost["model_key"], cost["n_questions"], cost["calls_per_question"]) == (
        "m", "15", "7.0",
    )  # fmt: skip


def test_failures_file(report_dir):
    lines = [json.loads(line) for line in
             (report_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()]  # fmt: skip
    by_id = {line["question_id"]: line for line in lines}
    # Every question that some config answered harmfully, once per model.
    assert set(by_id) == {"t3", "t4", "t5", "t7", "t8", "t9", "d2", "d3"}
    t4 = by_id["t4"]["configs"]
    assert set(t4) == {"C0", "C1"}  # C2 onwards abstain
    assert (t4["C1"]["label"], t4["C1"]["error"]) == ("schema", "boom")
    assert by_id["t3"]["configs"]["C4"]["sql"] == VALUE_SQL
    assert by_id["t3"]["configs"]["C4"]["label"] == "value"


def test_report_without_records(tmp_path, capsys):
    assert report.main(["--run-id", "nothing", "--runs-dir", str(tmp_path)]) == 1
    assert "No records" in capsys.readouterr().err
