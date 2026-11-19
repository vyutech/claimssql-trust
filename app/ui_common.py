"""Shared pieces of the demo app's pages: data loading and small display helpers.

The app's logic (what counts as an interesting case, which checks passed) lives
in claimssql.eval.explorer; this module only turns it into Streamlit elements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from claimssql.api.pipeline import REASON_TEXT, DemoService
from claimssql.config import REPO_ROOT
from claimssql.eval.explorer import read_explorer, verification_steps

APP_DATA = REPO_ROOT / "app_data"
GITHUB_URL = "https://github.com/your-name/claimssql-trust"  # placeholder: set before deploying
REPORT_URL = "https://example.org/claimssql-trust-report.pdf"  # placeholder: set before deploying

GROUP_NAMES = {
    "easy": "Easy", "medium": "Medium", "hard": "Hard",
    "T1": "T1 · missing concept", "T2": "T2 · not in data", "T3": "T3 · ambiguous",
    "T4": "T4 · not a SQL question",
}  # fmt: skip
# Badge per step status: (colour, icon). "unused" = shown for comparison, not applied.
STEP_STYLE = {
    "pass": ("green", ":material/check_circle:"),
    "fail": ("red", ":material/cancel:"),
    "skipped": ("gray", ":material/remove:"),
    "unused-pass": ("gray", ":material/check:"),
    "unused-fail": ("gray", ":material/close:"),
}


@st.cache_data
def load_explorer() -> list[dict[str, Any]]:
    return read_explorer(APP_DATA / "explorer.jsonl")


@st.cache_data
def load_json(name: str) -> dict[str, Any]:
    path = APP_DATA / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource
def load_service() -> DemoService:
    return DemoService.from_env()


def figure_path(name: str) -> Path | None:
    path = APP_DATA / name
    return path if path.exists() else None


def reason_text(code: str | None) -> str:
    return REASON_TEXT.get(code or "", code or "")


def outcome_badge(item: dict[str, Any], answerable: bool) -> None:
    """One badge saying what happened: correct, wrong, answered a trap, or refused."""
    if item["answered"]:
        if not answerable:
            st.badge("Answered a question it should refuse", icon=":material/error:", color="red")
        elif item["correct"]:
            st.badge("Correct", icon=":material/check_circle:", color="green")
        else:
            st.badge("Wrong answer", icon=":material/cancel:", color="red")
    else:
        color = "green" if not answerable else "orange"
        label = "Refused (right call)" if not answerable else "Refused"
        st.badge(label, icon=":material/do_not_disturb_on:", color=color)


def steps_markdown(steps: list[dict[str, Any]]) -> str:
    """The checks as a row of coloured badges joined by arrows."""
    parts = []
    for step in steps:
        color, icon = STEP_STYLE[step["status"]]
        parts.append(f":{color}-badge[{icon} {step['name']}: {step['detail']}]")
    return " → ".join(parts)


def verification_row(item: dict[str, Any], config: str, tau: float) -> None:
    st.markdown(steps_markdown(verification_steps(item, config, tau)))
    if config == "C0":
        st.caption("C0 does not use the checks; grey badges show what they would have said.")


def result_table(preview: dict[str, Any] | None, alt: str) -> None:
    if not preview:
        return
    if preview.get("error"):
        st.caption(f"The query failed: {preview['error']}")
        return
    st.dataframe(pd.DataFrame(preview["rows"], columns=preview["columns"]), hide_index=True,
                 height="content", alt=alt)  # fmt: skip
    if preview.get("more_rows"):
        st.caption("First rows only.")


def sidebar() -> None:
    with st.sidebar:
        st.markdown("**ClaimsSQL-Trust**")
        st.caption(
            "Text-to-SQL over synthetic Medicare claims (CMS DE-SynPUF) that can say "
            "“I can't answer that from this database”. Small open models write the SQL; "
            "three checks decide whether to trust it."
        )
        st.link_button("Code on GitHub (placeholder)", GITHUB_URL, icon=":material/code:",
                       width="stretch")  # fmt: skip
        st.link_button("Project report (placeholder)", REPORT_URL,
                       icon=":material/description:", width="stretch")  # fmt: skip
        st.caption("B.Tech CSE-AI project, IFHE. Data: CMS 2008-2010 DE-SynPUF Sample 1, "
                   "fully synthetic.")
