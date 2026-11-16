"""ClaimsSQL-Trust demo app.

Two modes:
  Benchmark explorer     what each model did on each benchmark question, from
                         the precomputed app_data/explorer.jsonl (no model calls)
  Ask your own question  the live pipeline on the 5,000-member demo database

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable without installing it (needed on Hugging Face).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from claimssql.api.pipeline import REASON_TEXT, DemoService  # noqa: E402
from claimssql.config import REPO_ROOT  # noqa: E402
from claimssql.eval.explorer import read_explorer  # noqa: E402
from claimssql.llm.base import LLMError, RateLimitStop  # noqa: E402

EXPLORER_FILE = REPO_ROOT / "app_data" / "explorer.jsonl"
CONFIG_NAMES = {
    "C0": "C0: no verification (the model must answer)",
    "C4": "C4: full verification (the model may refuse; V1, V2 and V3 must pass)",
}
GROUP_NAMES = {
    "easy": "easy", "medium": "medium", "hard": "hard",
    "T1": "T1 trap: missing concept", "T2": "T2 trap: not in data",
    "T3": "T3 trap: ambiguous", "T4": "T4 trap: not a SQL question",
}  # fmt: skip

st.set_page_config(page_title="ClaimsSQL-Trust", page_icon="🩺", layout="wide")


@st.cache_data
def load_explorer() -> list[dict]:
    return read_explorer(EXPLORER_FILE)


@st.cache_resource
def load_service() -> DemoService:
    return DemoService.from_env()


def table(preview: dict | None) -> None:
    if not preview:
        return
    if preview.get("error"):
        st.error(f"Error: {preview['error']}")
        return
    st.dataframe(pd.DataFrame(preview["rows"], columns=preview["columns"]),
                 hide_index=True, width="stretch")  # fmt: skip
    if preview.get("more_rows"):
        st.caption("First rows only.")


def show_config(name: str, item: dict, answerable: bool) -> None:
    st.markdown(f"**{CONFIG_NAMES[name]}**")
    if item["answered"]:
        if not answerable:
            st.error("Answered a question that cannot be answered from this database.")
        elif item["correct"]:
            st.success("Answered, and the result matches the gold answer.")
        else:
            st.error("Answered, but the result does not match the gold answer.")
        st.code(item["sql"], language="sql")
        table(item["result"])
    else:
        message = REASON_TEXT.get(item["reason_code"], item["reason_code"])
        good = " That is the right call: there is no answer here." if not answerable else ""
        st.warning(f"Abstained: {message}{good}")
        if item.get("model_sql"):
            st.caption("SQL the model wrote (not used):")
            st.code(item["model_sql"], language="sql")

    check = item["verification"]
    with st.expander("How this was verified"):
        unknown = check["unknown_tables"] + check["unknown_columns"]
        st.markdown("**V1 schema check:** "
                    + ("unknown names: " + ", ".join(f"`{u}`" for u in unknown) if unknown
                       else f"parse error: {check['parse_error']}" if check["parse_error"]
                       else "every table and column exists"))  # fmt: skip
        st.markdown("**V2 dry run:** " + (f"{check['exec_error_class']} error - "
                                          f"{check['exec_error']}" if check["exec_error_class"]
                                          else "ran without errors"))  # fmt: skip
        if name == "C4" and check["agreement"] is not None:
            st.markdown(f"**V3 self-consistency:** agreement {check['agreement']:.0%} of "
                        f"{check['k']} samples; result groups {check['cluster_sizes'] or 'none'}")
        elif name == "C0":
            st.markdown("**V3 self-consistency:** not used by C0")


def explorer() -> None:
    rows = load_explorer()
    if not rows:
        st.info("No explorer data yet. Build it with `python scripts/build_demo_db.py`.")
        return
    models = sorted({r["model_key"] for r in rows})
    left, right = st.columns([1, 3])
    model = left.selectbox("Model", models)
    groups = ["all", *GROUP_NAMES]
    group = left.selectbox("Question group", groups,
                           format_func=lambda g: GROUP_NAMES.get(g, "all groups"))  # fmt: skip
    choices = [r for r in rows if r["model_key"] == model and group in ("all", r["group"])]
    row = right.selectbox("Question", choices,
                          format_func=lambda r: f"{r['question_id']}  {r['question'][:110]}")
    if row is None:
        return

    st.subheader(row["question"])
    st.caption(f"{row['question_id']} · {GROUP_NAMES.get(row['group'], row['group'])} · "
               f"{row['split']} split · model {row['model']} · C4 threshold τ = {row['tau']:g}")
    if row["answerable"]:
        with st.expander("Gold answer", expanded=True):
            st.code(row["gold_sql"], language="sql")
            table(row["gold"])
    else:
        st.info("This question cannot be answered from the database. "
                + (f"Why: {row['notes']}" if row["notes"] else ""))
    c0, c4 = st.columns(2)
    with c0:
        show_config("C0", row["configs"]["C0"], row["answerable"])
    with c4:
        show_config("C4", row["configs"]["C4"], row["answerable"])


def ask_mode() -> None:
    st.write("Ask a question about the claims data. The system answers only if the SQL "
             "passes every check; otherwise it says why it won't answer.")
    st.caption("Runs on a 5,000-member subset of CMS DE-SynPUF Sample 1 (synthetic data), so "
               "numbers differ from the full benchmark. One question takes 1-2 minutes on the "
               "free model tier. The demo answers when at least 2 of 3 extra samples agree "
               "(a demo setting, not a benchmark result).")
    question = st.text_area("Your question", max_chars=1000,
                            placeholder="How many members had diabetes in 2009?")
    if not st.button("Ask", type="primary", disabled=not question.strip()):
        return
    try:
        service = load_service()
    except Exception as exc:  # noqa: BLE001 - shown to the user instead of a stack trace
        st.error(f"The demo is not set up: {exc}")
        return
    try:
        with st.status("Working on it...", expanded=True) as status:
            answer = service.ask(question.strip(), progress=status.write)
            status.update(label="Done", state="complete", expanded=False)
    except RateLimitStop:
        st.error("The free daily limit of the model provider has been reached. Please try "
                 "again tomorrow, or use the Benchmark explorer, which needs no model calls.")
        return
    except LLMError as exc:
        st.error(f"The model could not be reached: {exc}")
        return

    if answer.answered:
        st.success("Answered")
        st.code(answer.sql, language="sql")
        st.dataframe(pd.DataFrame(answer.rows, columns=answer.columns), hide_index=True,
                     width="stretch")  # fmt: skip
        if answer.more_rows:
            st.caption("First 100 rows only.")
    else:
        st.warning(f"Abstained: {answer.reason_text}")
        if answer.model_sql:
            st.caption("SQL the model wrote (not used):")
            st.code(answer.model_sql, language="sql")
    with st.expander("How this was verified"):
        unknown = answer.unknown_tables + answer.unknown_columns
        st.markdown("**V1 schema check:** " + (", ".join(f"`{u}`" for u in unknown)
                                               if unknown else "no unknown names"))
        st.markdown("**V2 dry run:** " + (answer.exec_error or "ran without errors"))
        if answer.agreement is not None:
            st.markdown(f"**V3 self-consistency:** agreement {answer.agreement:.0%} of "
                        f"{answer.k} samples (needs {answer.tau:.0%})")
        st.caption(f"Model {answer.model} · {answer.calls} calls "
                   f"({answer.cached_calls} from the cache)")


st.title("ClaimsSQL-Trust")
st.write("Text-to-SQL over Medicare claims that can say "
         "**“I can't answer that from this database.”**")
mode = st.sidebar.radio("Mode", ["Benchmark explorer", "Ask your own question"])
if mode == "Benchmark explorer":
    explorer()
else:
    ask_mode()
