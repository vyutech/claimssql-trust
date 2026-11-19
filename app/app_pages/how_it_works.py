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

st.space("small")
with st.expander("Glossary", icon=":material/menu_book:"):
    st.markdown(
        "- **Hallucination**: the model produces SQL or an answer that looks plausible but "
        "is wrong, such as an invented column or a misread code.\n"
        "- **Abstention (refusal)**: the system declines to answer and gives a reason, "
        "instead of guessing.\n"
        "- **C0 to C4**: five ways of running the same model. C0 must always answer; C1 "
        "lets the model refuse; C2 adds check V1; C3 adds V2; C4 adds V3.\n"
        "- **Harmful answer**: a wrong answer to a question that has one, or any answer to a "
        "question the data cannot answer.\n"
        "- **Reliability Score (RS₁₀)**: +1 for a right answer or a justified refusal, 0 for "
        "an unneeded refusal, −10 for a harmful answer; times 100 and averaged. A wrong "
        "answer costs ten right ones.\n"
        "- **Coverage**: the share of questions the system answers at all."
    )
