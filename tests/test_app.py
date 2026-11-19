"""P7C tests: the app's explorer logic, its data files, and the pages (AppTest).

The pages are run headless with Streamlit's AppTest on small hand-made app
data, so no model, database or network is needed.
"""

import json
import random
import shutil
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


@pytest.fixture()
def app_data(tmp_path, monkeypatch):
    """Point the app at small hand-made data files."""
    import ui_common

    folder = tmp_path / "app_data"
    folder.mkdir()
    (folder / "explorer.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS),
                                           encoding="utf-8")  # fmt: skip
    (folder / "results.json").write_text(json.dumps({"n_test": 10, "models": [
        {"model_key": "a", "model": "a-tag", "tau": 0.6, "harmful_c0": 8.0, "harmful_c4": 2.0,
         "harmful_c4_lo": 0.0, "harmful_c4_hi": 4.0, "rs10_c0": -300.0, "rs10_c4": 10.0,
         "coverage_c4": 0.3, "ex_c0": 0.5, "ex_c4": 0.3},
    ]}), encoding="utf-8")  # fmt: skip
    (folder / "samples.json").write_text(json.dumps({
        "Should answer": [{"id": "E-9", "group": "easy", "question": "sample answerable"}],
        "Should refuse": [{"id": "T1-9", "group": "T1", "question": "sample refusable"}],
    }), encoding="utf-8")  # fmt: skip
    shutil.copy(ROOT / "results" / "figures" / "fig7_2_harmful_by_config.png", folder)
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


def test_results_page(app_data):
    at = run_page()
    assert at.title[0].value == "Can a small model know when not to answer?"
    metrics = {m.label: (m.value, m.delta) for m in at.metric}
    assert metrics["Harmful answers, C0 → C4"] == ("8 → 2", "-75%")
    assert metrics["Reliability Score RS₁₀, C0 → C4"] == ("-300 → 10", "+310")
    assert "**8–8** got a harmful answer" in at.markdown[0].value


def test_explorer_page_and_filters(app_data):
    at = run_page("app_pages/explorer.py")
    assert at.subheader[0].value == "question E-1"
    # st.badge is recorded as Markdown (":blue-badge[Easy]") in AppTest.
    badges = " | ".join(m.value for m in at.markdown if "-badge[" in m.value)
    assert ":blue-badge[Easy]" in badges and ":gray-badge[test split]" in badges
    assert "payment type" in badges
    assert "Wrong answer]" in badges and "Refused]" in badges  # C0 wrong, C4 refused

    at.toggle(key="saved_only").set_value(True).run()
    assert not at.exception
    assert [o for o in at.selectbox[-1].options] == ["E-1 · question E-1", "T1-1 · question T1-1"]

    at.button(key="interesting").click().run()
    assert not at.exception
    assert at.session_state["saved_only"] is True
    assert at.subheader[0].value in ("question E-1", "question T1-1")


def test_explorer_page_without_data(tmp_path, monkeypatch):
    import ui_common

    monkeypatch.setattr(ui_common, "APP_DATA", tmp_path)
    at = run_page("app_pages/explorer.py")
    assert "No explorer data yet" in at.info[0].value


def test_ask_page_shows_samples_without_calling_a_model(app_data):
    at = run_page("app_pages/ask.py")
    labels = [b.label for b in at.button]
    assert "sample answerable" in labels and "sample refusable" in labels
    assert at.session_state["answer"] is None  # nothing ran on its own
