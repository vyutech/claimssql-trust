"""decide(): every configuration and every reason code, on hand-made records."""

import pytest

from claimssql.policy.decide import (
    CONFIGS,
    EXEC_ERROR,
    LOW_AGREEMENT,
    MODEL_REFUSED,
    PARSE_ERROR,
    REASON_CODES,
    SCHEMA_UNKNOWN,
    Decision,
    decide,
)

FORCED_SQL = "SELECT 1 -- forced"
ABSTAIN_SQL = "SELECT 2 -- abstain"
TAU = 0.6


def answer(sql: str) -> dict:
    return {"parsed": {"answerable": True, "sql": sql, "confidence": 90, "reason": ""},
            "error": None}  # fmt: skip


REFUSAL = {"parsed": {"answerable": False, "sql": None, "confidence": 90, "reason": "no"},
           "error": None}  # fmt: skip
UNREADABLE = {"parsed": None, "error": "PARSE_ERROR"}


def record(**overrides) -> dict:
    """A record where every check passes; tests break one thing at a time."""
    base = {
        "forced": answer(FORCED_SQL),
        "abstain": answer(ABSTAIN_SQL),
        "v1": {"passed": True, "unknown_tables": [], "unknown_columns": [], "parse_error": None},
        "v2": {"ok": True, "error_class": None, "result_hash": "h"},
        "v3": {"agreement": 0.8, "greedy_in_majority": True},
    }
    return {**base, **overrides}


def outcome(rec: dict) -> dict[str, tuple]:
    return {name: tuple(decide(rec, name, TAU)) for name in CONFIGS}


def test_everything_passes_every_config_answers():
    assert outcome(record()) == {
        "C0": (True, FORCED_SQL, None),
        "C1": (True, ABSTAIN_SQL, None),
        "C2": (True, ABSTAIN_SQL, None),
        "C3": (True, ABSTAIN_SQL, None),
        "C4": (True, ABSTAIN_SQL, None),
    }


def test_model_refusal_under_the_abstain_prompt():
    got = outcome(record(abstain=REFUSAL))
    assert got["C0"] == (True, FORCED_SQL, None)  # C0 uses the forced generation
    for name in ("C1", "C2", "C3", "C4"):
        assert got[name] == (False, None, MODEL_REFUSED)


def test_c0_abstains_when_the_forced_model_refuses_or_is_unreadable():
    assert outcome(record(forced=REFUSAL))["C0"] == (False, None, MODEL_REFUSED)
    assert outcome(record(forced=UNREADABLE))["C0"] == (False, None, PARSE_ERROR)
    # ... and that does not affect the other configs.
    assert outcome(record(forced=REFUSAL))["C1"] == (True, ABSTAIN_SQL, None)


def test_unreadable_reply_is_parse_error_from_c1_on():
    got = outcome(record(abstain=UNREADABLE))
    assert all(got[name] == (False, None, PARSE_ERROR) for name in ("C1", "C2", "C3", "C4"))
    assert outcome(record(abstain=None))["C1"] == (False, None, PARSE_ERROR)


def test_answerable_without_sql_counts_as_refusal():
    empty = {"parsed": {"answerable": True, "sql": "  ", "confidence": 50}, "error": None}
    assert outcome(record(abstain=empty))["C1"] == (False, None, MODEL_REFUSED)


def test_unknown_identifier_stops_c2_and_later():
    v1 = {"passed": False, "unknown_tables": [], "unknown_columns": ["AGE"], "parse_error": None}
    got = outcome(record(v1=v1))
    assert got["C1"] == (True, ABSTAIN_SQL, None)  # C1 does not look at V1
    assert all(got[name] == (False, None, SCHEMA_UNKNOWN) for name in ("C2", "C3", "C4"))


def test_unparseable_sql_at_v1_is_exec_error_not_schema_unknown():
    v1 = {"passed": False, "unknown_tables": [], "unknown_columns": [], "parse_error": "bad"}
    got = outcome(record(v1=v1))
    assert got["C1"][0] is True
    assert all(got[name] == (False, None, EXEC_ERROR) for name in ("C2", "C3", "C4"))


def test_execution_failure_stops_c3_and_later():
    got = outcome(record(v2={"ok": False, "error_class": "binder", "error": "x"}))
    assert got["C1"][0] and got["C2"][0]
    assert got["C3"] == got["C4"] == (False, None, EXEC_ERROR)


def test_schema_error_is_reported_before_execution_error():
    v1 = {"passed": False, "unknown_tables": ["t"], "unknown_columns": [], "parse_error": None}
    got = outcome(record(v1=v1, v2={"ok": False, "error_class": "catalog"}))
    assert got["C3"] == (False, None, SCHEMA_UNKNOWN)


def test_low_agreement_stops_only_c4():
    got = outcome(record(v3={"agreement": 0.4, "greedy_in_majority": True}))
    assert got["C3"] == (True, ABSTAIN_SQL, None)
    assert got["C4"] == (False, None, LOW_AGREEMENT)
    # High agreement among the samples, but the greedy answer disagrees with them.
    got = outcome(record(v3={"agreement": 1.0, "greedy_in_majority": False}))
    assert got["C4"] == (False, None, LOW_AGREEMENT)


@pytest.mark.parametrize(
    ("agreement", "tau", "answered"),
    [(0.6, 0.6, True), (0.59, 0.6, False), (0.2, 0.0, True), (1.0, 1.0, True), (0.8, 1.0, False)],
)
def test_tau_boundary_is_inclusive(agreement, tau, answered):
    rec = record(v3={"agreement": agreement, "greedy_in_majority": True})
    assert decide(rec, "C4", tau).answered is answered


def test_missing_verifier_output_means_abstain():
    assert decide(record(v1=None), "C2").reason_code == SCHEMA_UNKNOWN
    assert decide(record(v2=None), "C3").reason_code == EXEC_ERROR
    assert decide(record(v3=None), "C4", TAU).reason_code == LOW_AGREEMENT


def test_bad_arguments():
    with pytest.raises(ValueError, match="C4 needs tau"):
        decide(record(), "C4")
    with pytest.raises(ValueError, match="unknown config"):
        decide(record(), "C5", TAU)
    assert decide(record(), "C3").answered  # tau is only needed for C4


def test_decision_shape_and_every_reason_code_is_reachable():
    decision = decide(record(), "C1")
    assert isinstance(decision, Decision)
    answered, sql, reason_code = decision  # unpacks as the 3-tuple from the spec
    assert (answered, sql, reason_code) == (True, ABSTAIN_SQL, None)

    v1_bad = {"passed": False, "unknown_columns": ["x"], "parse_error": None}
    seen = {
        decide(record(abstain=REFUSAL), "C4", TAU).reason_code,
        decide(record(abstain=UNREADABLE), "C4", TAU).reason_code,
        decide(record(v1=v1_bad), "C4", TAU).reason_code,
        decide(record(v2={"ok": False}), "C4", TAU).reason_code,
        decide(record(v3={"agreement": 0.0, "greedy_in_majority": False}), "C4", TAU).reason_code,
    }
    assert seen == set(REASON_CODES)
