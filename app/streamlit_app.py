"""ClaimsSQL-Trust demo app (entry point).

Pages, in app/app_pages/:
  Results    headline results of the benchmark and how the system works (default)
  Explorer   what each model did on each benchmark question, with the checks
  Ask        the live pipeline on the 5,000-member demo database

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable without installing it (needed on Hugging Face).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from ui_common import sidebar  # noqa: E402

st.set_page_config(page_title="ClaimsSQL-Trust", page_icon=":material/fact_check:",
                   layout="wide")  # fmt: skip

page = st.navigation(
    [
        st.Page("app_pages/results.py", title="Results", icon=":material/insights:",
                default=True,
                help="The benchmark's headline results: how many harmful answers each model "
                     "gives without and with the checks, and how the system works."),
        st.Page("app_pages/explorer.py", title="Explorer", icon=":material/manage_search:",
                help="Browse every benchmark question: what each model answered with and "
                     "without the checks, and which check stopped a bad answer."),
        st.Page("app_pages/ask.py", title="Ask", icon=":material/chat:",
                help="Ask your own question. The live pipeline answers only if every check "
                     "passes, on a 5,000-member subset of the data."),
    ],
    position="top",
)  # fmt: skip
sidebar()
page.run()
