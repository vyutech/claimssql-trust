"""How it works: the problem, the three checks with real examples, one question
walked through, and a glossary."""

import streamlit as st

from ui_common import load_json, reason_text

examples = load_json("examples.json")

st.title("How it works")
st.markdown(
    "Language models are good at writing SQL that *looks* right. On a real claims database "
    "they also invent column names, misread codes, or answer questions the data cannot "
    "answer, and they sound just as confident when they do. This system keeps the model, "
    "but checks every query before showing an answer, and **refuses** when a check fails."
)

st.subheader("The three safety checks")
cards = [
    ("v1", ":material/table_view:", "V1 · Does every table and column exist?",
     "The SQL is parsed and every name is looked up in the database schema. An invented "
     "column is caught before anything runs."),
    ("v2", ":material/play_circle:", "V2 · Does the SQL run?",
     "The query runs on a read-only copy with a time limit. Anything that errors out is "
     "refused."),
    ("v3", ":material/groups:", "V3 · Do 5 attempts agree?",
     "The model answers 5 more times with a little randomness. If their results disagree, "
     "the model is guessing, and the answer is refused."),
]  # fmt: skip
for column, (key, icon, title, text) in zip(st.columns(3, gap="medium"), cards, strict=True):
    with column.container(key=f"step_{key}", height="stretch"):
        st.markdown(f"## {icon}")
        st.markdown(f"**{title}**")
        st.markdown(text)
        example = examples.get(key)
        if key == "v3":
            example = examples.get("walkthrough")
        if not example:
            continue
        st.caption(f"Real example · {example['model']} · {example['question_id']}")
        st.markdown(f"*{example['question']}*")
        check = example["verification"]
        if key == "v1":
            names = ", ".join(check["unknown_tables"] + check["unknown_columns"])
            st.code(example["sql"], language="sql", wrap_lines=True)
            st.markdown(
                f":orange-badge[:material/do_not_disturb_on: Refused] `{names}` does not exist."
            )
        elif key == "v2":
            error = (check.get("exec_error") or "").split("\n")[0]
            st.code(example["sql"], language="sql", wrap_lines=True)
            st.markdown(
                f":orange-badge[:material/do_not_disturb_on: Refused] "
                f"{check['exec_error_class']} error"
            )
            st.caption(error[:180])
        else:
            st.markdown(
                f":orange-badge[:material/do_not_disturb_on: Refused] only "
                f"{check['agreement']:.0%} of the attempts agreed "
                f"(groups of {', '.join(str(s) for s in check['cluster_sizes'])})."
            )

walk = examples.get("walkthrough")
if walk:
    st.subheader("One question, followed all the way")
    st.caption(f"{walk['model']} on benchmark question {walk['question_id']}")
    gold = walk["gold_result"]["rows"][0][0] if walk.get("gold_result") else None
    c0 = walk["c0"]["result"]["rows"][0][0] if walk["c0"].get("result") else None
    check = walk["verification"]
    strip = st.columns(5, gap="small")
    with strip[0].container(key="step_w1", height="stretch"):
        st.markdown("**1 · Question**")
        st.markdown(walk["question"])
    with strip[1].container(key="step_w2", height="stretch"):
        st.markdown("**2 · The model's SQL**")
        st.code(walk["sql"], language="sql", wrap_lines=True)
    with strip[2].container(key="step_w3", height="stretch"):
        st.markdown("**3 · Why it is wrong**")
        st.markdown(walk["owner_note"])
        if c0 is not None and gold is not None:
            st.markdown(
                f":red-badge[Returns {c0:,}] every member, not the "
                f":green-badge[{gold:,}] who really have 3 or more."
            )
    with strip[3].container(key="step_w4", height="stretch"):
        st.markdown("**4 · The checks**")
        st.markdown(":green-badge[:material/check_circle: V1 · names exist]")
        st.markdown(":green-badge[:material/check_circle: V2 · it runs]")
        st.markdown(f":red-badge[:material/cancel: V3 · {check['agreement']:.0%} agree]")
        st.caption("The SQL is valid, so only the agreement check can catch this mistake.")
    with strip[4].container(key="step_w5", height="stretch"):
        st.markdown("**5 · The decision**")
        st.markdown("No checks (C0)  \n:red-badge[:material/cancel: Wrong answer]")
        st.markdown("All checks (C4)  \n:orange-badge[:material/do_not_disturb_on: Refused]")
        st.caption(reason_text(walk["reason_code"]))


# ------------------------------------------------------------- glossary ----
# Compact term cards in a two-column grid, grouped under three headings. The coloured
# dots follow the app's meaning: red = harmful, amber = refused, green = correct.
def term(key: str, name: str, text: str, example: str = "", dot: str = "") -> None:
    with st.container(key=f"term_{key}", height="stretch"):
        st.markdown(f":{dot}[●] **{name}**" if dot else f"**{name}**")
        st.markdown(text)
        if example:
            st.caption(example)


def grid(terms: list[tuple]) -> None:
    columns = st.columns(2, gap="small")
    for position, args in enumerate(terms):
        with columns[position % 2]:
            term(*args)


v1 = examples.get("v1")
invented = ", ".join(v1["verification"]["unknown_columns"]) if v1 else ""
coverage = [m["coverage_c4"] for m in load_json("results.json").get("models", [])]

st.space("small")
with st.expander("Glossary", icon=":material/menu_book:"):
    st.caption("THE PROBLEM")
    grid([
        ("hallucination", "Hallucination",
         "The model makes something up, such as a column or a code meaning, that looks "
         "real but is not.",
         f"Example: a column called {invented}, which does not exist." if invented else ""),
        ("harmful", "Harmful answer",
         "A wrong answer to a real question, or any answer to a trap question.",
         "Example: 112,754 members when the right number is 35,857.", "red"),
        ("trap", "Trap question (T1–T4)",
         "A question the data cannot answer, so the right response is to refuse.",
         "T1 missing concept · T2 not in the data · T3 ambiguous · T4 not a database "
         "question"),
    ])  # fmt: skip

    st.caption("WHAT THE SYSTEM DOES")
    grid([
        ("text_to_sql", "Text-to-SQL",
         "Turning a question in plain English into a database query that answers it."),
        ("abstention", "Abstention (refusal)",
         "The system says “I can't answer that” and gives the reason, instead of guessing.",
         "Reasons: the model refused, a name does not exist, the query fails, or the "
         "attempts disagree.", "orange"),
        ("agreement", "Self-consistency (agreement)",
         "The model answers 5 more times; if the results do not match, the answer is not "
         "trusted.",
         "Agreement = the share of attempts that gave the most common result."),
    ])  # fmt: skip

    st.caption("HOW IT'S MEASURED")
    grid([
        ("gold", "Gold answer",
         "The correct result, from a query the author wrote and checked by running it.",
         "", "green"),
        ("coverage", "Coverage",
         "The share of questions the system answers instead of refusing.",
         f"With all checks: {min(coverage):.0%}–{max(coverage):.0%} of test questions."
         if coverage else ""),
    ])  # fmt: skip
    with st.container(key="term_ladder"):
        st.markdown("**C0–C4: five ways to run the same model**")
        st.markdown(
            ":gray-badge[C0 · no checks] → :blue-badge[C1 · may refuse] → "
            ":blue-badge[C2 · + V1 names exist] → :blue-badge[C3 · + V2 query runs] → "
            ":blue-badge[C4 · + V3 attempts agree]"
        )
        st.caption("Each step keeps everything before it and adds one more check.")
    with st.container(key="term_rs"):
        st.markdown("**Reliability Score (RS₁₀)**")
        st.markdown(
            "Rewards right answers and safe refusals, and punishes a harmful answer ten "
            "times as hard. Worked example with 10 questions:"
        )
        st.markdown(
            "| Outcome | Questions | Points |\n|---|---:|---:|\n"
            "| Right answer | 6 | :green[+6] |\n"
            "| Justified refusal | 2 | :orange[+2] |\n"
            "| Unneeded refusal | 1 | 0 |\n"
            "| Wrong answer | 1 | :red[−10] |"
        )
        st.markdown("(6 + 2 + 0 − 10) / 10 × 100 = **−20**")
        st.caption("One wrong answer cancels out eight good ones.")
