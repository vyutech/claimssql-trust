"""P7C/P7D tests: the app's explorer logic, its data files, and the pages (AppTest).

The pages are run headless with Streamlit's AppTest on small hand-made app
data, so no model, database or network is needed.
"""

import json
import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from claimssql.eval import explorer as ex

ROOT = Path(__file__).parent.parent
APP = ROOT / "app" / "streamlit_app.py"


def item(answered, correct, region, reason=None, sql="SELECT 1", unknown=(), exec_error=None,
         agreement=1.0, in_majority=True):  # fmt: skip
    return {
        "answered": answered, "correct": correct, "region": region, "reason_code": reason,
        "sql": sql if answered else None, "model_sql": sql, "result": None,
        "verification": {"unknown_tables": [], "unknown_columns": list(unknown),
                         "parse_error": None, "exec_error_class": exec_error, "exec_error": None,
                         "agreement": agreement, "cluster_sizes": [5], "k": 5,
                         "greedy_in_majority": in_majority},
    }  # fmt: skip


def row(qid, model, group, c0, c4, tags=()):
    return {"model_key": model, "model": f"{model}-tag", "question_id": qid,
            "question": f"question {qid}", "group": group, "split": "test",
            "answerable": group in ("easy", "medium", "hard"), "notes": "", "gold_sql": None,
            "gold": None, "tau": 0.6, "trap_tags": list(tags), "configs": {"C0": c0, "C4": c4}}


SAVED_WRONG = row("E-1", "a", "easy", item(True, False, "answerable_wrong"),
                  item(False, False, "answerable_abstained", "LOW_AGREEMENT", agreement=0.2),
                  tags=("payment_type",))  # fmt: skip
SAVED_TRAP = row("T1-1", "a", "T1", item(True, False, "infeasible_answered"),
                 item(False, False, "infeasible_abstained", "MODEL_REFUSED", sql=None))
BOTH_RIGHT = row("E-2", "a", "easy", item(True, True, "answerable_correct"),
                 item(True, True, "answerable_correct"))
OTHER_MODEL = row("E-1", "b", "easy", item(True, False, "answerable_wrong"),
                  item(True, False, "answerable_wrong"))
ROWS = [SAVED_WRONG, SAVED_TRAP, BOTH_RIGHT, OTHER_MODEL]


# --------------------------------------------------------------- logic ----


def test_saved_by_checks():
    assert [ex.saved_by_checks(r) for r in ROWS] == [True, True, False, False]


def test_filter_rows():
    ids = lambda rows: [(r["model_key"], r["question_id"]) for r in rows]  # noqa: E731
    assert ids(ex.filter_rows(ROWS, model="a")) == [("a", "E-1"), ("a", "T1-1"), ("a", "E-2")]
    assert ids(ex.filter_rows(ROWS, groups=["T1"])) == [("a", "T1-1")]
    assert ids(ex.filter_rows(ROWS, tag="payment_type")) == [("a", "E-1")]
    assert ids(ex.filter_rows(ROWS, model="a", saved_only=True)) == [("a", "E-1"), ("a", "T1-1")]
    assert ex.filter_rows(ROWS, model="b", saved_only=True) == []


def test_pick_interesting_prefers_saved_cases():
    rng = random.Random(0)
    picks = {ex.pick_interesting(ROWS, rng)["question_id"] for _ in range(30)}
    assert picks == {"E-1", "T1-1"} or picks <= {"E-1", "T1-1"}
    assert ex.pick_interesting([BOTH_RIGHT], rng) is BOTH_RIGHT  # no saved case: any case
    assert ex.pick_interesting([], rng) is None


def test_verification_steps():
    c4 = ex.verification_steps(SAVED_WRONG["configs"]["C4"], "C4", 0.6)
    assert [(s["name"], s["status"]) for s in c4] == [
        ("Model", "pass"), ("V1 schema", "pass"), ("V2 dry run", "pass"), ("V3 agreement", "fail"),
    ]  # fmt: skip
    assert c4[3]["detail"] == "20% agree, needs 60%"

    c0 = ex.verification_steps(SAVED_WRONG["configs"]["C0"], "C0", 0.6)
    assert [s["status"] for s in c0] == ["pass", "unused-pass", "unused-pass"]  # no V3 for C0

    refused = ex.verification_steps(SAVED_TRAP["configs"]["C4"], "C4", 0.6)
    assert [(s["status"], s["detail"]) for s in refused] == [
        ("fail", "gave no SQL"), ("skipped", "not run"), ("skipped", "not run"),
        ("skipped", "not run"),
    ]  # fmt: skip

    broken = item(False, False, "answerable_abstained", "SCHEMA_UNKNOWN",
                  unknown=["beneficiary.AGE"], exec_error="binder")  # fmt: skip
    steps = ex.verification_steps(broken, "C4", 0.6)
    assert (steps[1]["status"], steps[1]["detail"]) == ("fail", "unknown: beneficiary.AGE")
    assert (steps[2]["status"], steps[2]["detail"]) == ("fail", "binder")


def test_sample_questions_use_dev_only_and_the_three_groups():
    questions = [json.loads(line) for line in
                 (ROOT / "benchmark" / "questions.jsonl").read_text("utf-8").splitlines()]
    samples = ex.sample_questions(questions)
    assert list(samples) == list(ex.SAMPLE_GROUPS)
    by_id = {q["id"]: q for q in questions}
    for picked in samples.values():
        for sample in picked:
            original = by_id[sample["id"]]
            assert original["split"] == "dev"  # no test question is exposed
            assert sample["question"] == original["question"]  # owner's wording, unchanged
    assert [s["group"] for s in samples["Should refuse"]] == ["T1", "T2", "T3", "T4"]
    assert all(s["group"] == "easy" for s in samples["Should answer"])
    assert all(s["trap_tags"] for s in samples["Tricky (trap tags)"])


def test_results_summary_from_the_report_tables():
    summary = ex.results_summary(ROOT / "results" / "tables")
    assert summary["n_test"] == 160
    assert [m["model_key"] for m in summary["models"]] == ["qwen", "gemma", "mid_oss"]
    qwen = summary["models"][0]
    assert (qwen["harmful_c0"], qwen["harmful_c4"], qwen["tau"]) == (75.0, 4.0, 0.8)
    assert qwen["harmful_c4_lo"] <= qwen["harmful_c4"] <= qwen["harmful_c4_hi"]


# ------------------------------------------------------------- pages ----


def test_results_detail_from_the_report_tables():
    detail = ex.results_detail(ROOT / "results" / "tables")
    configs = [(h["model"], h["config"]) for h in detail["harmful"]]
    assert ("qwen", "C0") in configs and ("mid_oss", "C4") in configs
    assert all(h["lo"] <= h["harmful"] <= h["hi"] for h in detail["harmful"])
    assert {i["group"] for i in detail["infeasible"]} == {"T1", "T2", "T3", "T4"}
    assert all(0 <= i["answered_c4"] <= i["n"] for i in detail["infeasible"])
    assert {a["signal"] for a in detail["auroc"]} >= {"1 - agreement", "100 - confidence"}
    rq4 = next(d for d in detail["differences"]
               if d["comparison"] == "mid_oss C4 minus qwen C4")  # fmt: skip
    assert rq4["excludes_zero"] is False  # the approved RQ4 finding: not clearly different


def test_check_examples_pick_one_case_per_check():
    examples = ex.check_examples(ROWS + [
        row("E-3", "a", "easy", item(True, False, "answerable_wrong"),
            item(False, False, "answerable_abstained", "SCHEMA_UNKNOWN", sql="SELECT AGE FROM t",
                 unknown=["t.AGE"])),
        row("E-4", "a", "easy", item(True, False, "answerable_wrong"),
            item(False, False, "answerable_abstained", "EXEC_ERROR", sql="SELECT x",
                 exec_error="binder")),
    ], walkthrough=("a", "E-1"))  # fmt: skip
    assert examples["v1"]["question_id"] == "E-3"
    assert examples["v2"]["question_id"] == "E-4"
    walk = examples["walkthrough"]
    assert walk["question_id"] == "E-1" and walk["reason_code"] == "LOW_AGREEMENT"
    assert walk["c0"]["answered"] is True and walk["c0"]["correct"] is False


def model_summary(key, c0, c4):
    return {"model_key": key, "model": f"{key}-tag", "tau": 0.6, "harmful_c0": c0,
            "harmful_c4": c4, "harmful_c4_lo": 0.0, "harmful_c4_hi": c4 + 2, "rs10_c0": -300.0,
            "rs10_c4": 10.0, "coverage_c4": 0.3, "ex_c0": 0.5 if key == "a" else 0.4,
            "ex_c4": 0.3}  # fmt: skip


def difference(comparison, metric, value, excludes_zero=True):
    return {"comparison": comparison, "metric": metric, "difference": value, "lo": value - 1,
            "hi": value + 1, "excludes_zero": excludes_zero}  # fmt: skip


RESULTS = {
    "n_test": 10,
    "models": [model_summary("a", 8.0, 2.0), model_summary("b", 6.0, 3.0)],
    "harmful": [{"model": m, "config": c, "harmful": h, "lo": h - 1, "hi": h + 1}
                for m in ("a", "b") for c, h in (("C0", 8.0), ("C4", 2.0))],
    "infeasible": [{"model": m, "group": g, "n": 4, "answered_c0": 2, "answered_c4": 0}
                   for m in ("a", "b") for g in ("T1", "T3")],
    "auroc": [{"model": m, "signal": s, "auroc": v, "lo": v - 0.1, "hi": v + 0.1}
              for m in ("a", "b") for s, v in (("1 - agreement", 0.88), ("100 - confidence", 0.8))],
    "differences": [
        difference("a C4 minus a C0", "harmful", -6.0),
        difference("b C4 minus b C0", "harmful", -3.0),
        difference("a AUROC(1 - agreement) minus a AUROC(100 - confidence)", "auroc", 0.08),
        difference("b AUROC(1 - agreement) minus b AUROC(100 - confidence)", "auroc", 0.01,
                   excludes_zero=False),
        difference("b C4 minus a C0", "rs_10", 300.0),
        difference("a C4 minus b C4", "rs_10", 12.0, excludes_zero=False),
    ],
}  # fmt: skip


@pytest.fixture()
def app_data(tmp_path, monkeypatch):
    """Point the app at small hand-made data files."""
    import ui_common

    folder = tmp_path / "app_data"
    folder.mkdir()
    (folder / "explorer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS),
                                           encoding="utf-8")  # fmt: skip
    (folder / "results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    walk = {**ex.check_examples(ROWS, walkthrough=("a", "E-1"))["walkthrough"],
            "gold_result": {"columns": ["n"], "rows": [[35857]]},
            "owner_note": "Flags are 1 = yes, 2 = no."}  # fmt: skip
    walk["c0"]["result"] = {"columns": ["n"], "rows": [[112754]]}
    (folder / "examples.json").write_text(json.dumps({"v1": None, "v2": None,
                                                      "walkthrough": walk}), encoding="utf-8")
    (folder / "samples.json").write_text(json.dumps({
        "Should answer": [{"id": "E-9", "group": "easy", "question": "sample answerable"}],
        "Should refuse": [{"id": "T1-9", "group": "T1", "question": "sample refusable"}],
    }), encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(ui_common, "APP_DATA", folder)
    return folder


@pytest.fixture(autouse=True)
def app_on_path(monkeypatch):
    import streamlit as st

    monkeypatch.syspath_prepend(str(ROOT / "app"))
    st.cache_data.clear()  # each test brings its own app data
    yield
    st.cache_data.clear()


def run_page(page: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


def text(at: AppTest) -> str:
    return " | ".join(m.value for m in at.markdown)


def test_home_page(app_data):
    at = run_page()  # Home is the default page
    assert at.title[0].value == "ClaimsSQL-Trust"
    page = text(at)
    assert "knows when to say" in page
    assert "−75%" in page  # best cut: a 8 → 2 (b 6 → 3 is only −50%)
    assert "1 · Ask" in page and "2 · Check" in page and "3 · Answer or refuse" in page
    labels = [b.label for b in at.button]
    assert "See how it works" in labels and "Try it yourself" in labels


def test_how_it_works_page(app_data):
    at = run_page("app_pages/how_it_works.py")
    page = text(at)
    assert "V1 · Does every table and column exist?" in page
    assert "V3 · Do 5 attempts agree?" in page
    assert ":red-badge[Returns 112,754]" in page and ":green-badge[35,857]" in page
    assert "Flags are 1 = yes, 2 = no." in page
    assert at.expander[-1].label == "Glossary"
    glossary = at.expander[-1]
    headings = [c.value for c in glossary.caption]
    assert headings[0] == "THE PROBLEM" and "WHAT THE SYSTEM DOES" in headings
    assert "HOW IT'S MEASURED" in headings
    terms = " | ".join(m.value for m in glossary.markdown)
    for name in ("Hallucination", "Trap question (T1–T4)", "Text-to-SQL",
                 "Self-consistency (agreement)", "Coverage", "Reliability Score (RS₁₀)"):
        assert f"**{name}**" in terms, name
    # Colour dots: red = harmful, amber = refused, green = correct.
    assert ":red[●] **Harmful answer**" in terms
    assert ":orange[●] **Abstention (refusal)**" in terms
    assert ":green[●] **Gold answer**" in terms
    assert ":gray-badge[C0 · no checks] → :blue-badge[C1 · may refuse]" in terms
    assert ":blue-badge[C4 · + V3 attempts agree]" in terms
    assert "(6 + 2 + 0 − 10) / 10 × 100 = **−20**" in terms
    assert "One wrong answer cancels out eight good ones." in headings
    assert "With all checks: 30%–30% of test questions." in headings  # from results.json


def test_results_page(app_data):
    at = run_page("app_pages/results.py")
    assert at.title[0].value == "Results"
    page = text(at)
    assert "**6–8 of 10**" in page  # RQ1: harmful answers at C0 across models
    assert "**8 → 2** for a-tag (−75%)" in page and "every interval excludes zero" in page
    assert "a-tag +0.08 (clearly better)" in page and "b-tag +0.01 (not clearly" in page
    assert "b-tag with checks vs. a-tag without: RS₁₀ **+300**" in page
    assert "not clearly different" in page
    assert len(at.get("vega_lite_chart")) == 2


def test_explorer_page_and_filters(app_data):
    at = run_page("app_pages/explorer.py")
    assert at.subheader[0].value == "question E-1"
    assert at.markdown[0].value.startswith("Every benchmark question, as each model")
    assert at.caption[0].value.startswith("**How to read this page**\n- Pick a model")
    assert ":green[●] correct · :red[●] wrong · :orange[●] refused" in at.caption[0].value
    assert "across all 3 questions" in at.caption[1].value  # model "a" has 3 rows
    assert at.toggle(key="saved_only").label == (
        "Only cases where the checks prevented a harmful answer")
    # st.badge is recorded as Markdown (":blue-badge[Easy]") in AppTest.
    badges = " | ".join(m.value for m in at.markdown if "-badge[" in m.value)
    assert ":blue-badge[Easy]" in badges and ":gray-badge[test split]" in badges
    assert "payment type" in badges
    assert ":red-badge[:material/cancel: Wrong answer]" in badges  # C0 wrong
    assert ":orange-badge[:material/do_not_disturb_on: Refused]" in badges  # C4 refused

    at.toggle(key="saved_only").set_value(True).run()
    assert not at.exception
    assert [o for o in at.selectbox[-1].options] == ["E-1 · question E-1", "T1-1 · question T1-1"]
    at.selectbox[-1].set_value(SAVED_TRAP).run()
    badges = " | ".join(m.value for m in at.markdown if "-badge[" in m.value)
    assert "Wrong: answered a trick question" in badges and "Refused safely" in badges

    at.button(key="interesting").click().run()
    assert not at.exception
    assert at.session_state["saved_only"] is True
    assert at.subheader[0].value in ("question E-1", "question T1-1")


def test_explorer_page_without_data(tmp_path, monkeypatch):
    import ui_common

    monkeypatch.setattr(ui_common, "APP_DATA", tmp_path)
    at = run_page("app_pages/explorer.py")
    assert "No explorer data yet" in at.info[0].value


def test_try_it_page_shows_samples_without_calling_a_model(app_data):
    at = run_page("app_pages/try_it.py")
    assert at.title[0].value == "Try it yourself"
    labels = [b.label for b in at.button]
    assert "sample answerable" in labels and "sample refusable" in labels
    assert at.session_state["answer"] is None  # nothing ran on its own


def test_about_page(app_data):
    at = run_page("app_pages/about.py")
    page = text(at)
    assert "Vyshnavi Vadla" in page and "fully synthetic" in page
    assert "only 30%–30% of" in page  # coverage comes from results.json
    assert len(at.dataframe[0].value) == 3  # the three benchmark models from models.yaml
    assert "IFHE" not in page


def test_no_sidebar_without_a_github_url(app_data):
    at = run_page()
    assert len(at.sidebar.children) == 0
