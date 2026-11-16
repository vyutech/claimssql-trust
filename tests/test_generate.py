"""P4 prompt, parsing and generator tests, with a fake client (no network)."""

import importlib.util
import json
from pathlib import Path

import pytest

from claimssql.config import load_experiment
from claimssql.generate import prompts
from claimssql.generate.generator import (
    DEGENERATE_OUTPUT,
    PARSE_ERROR,
    ParseFailure,
    clean_output,
    generate_all,
    generate_one,
    parse_output,
)
from claimssql.llm.base import DegenerateOutput
from claimssql.llm.cache import CachedClient, DiskCache
from fakes import GOOD, REFUSAL, FakeClient

ROOT = Path(__file__).parent.parent
CONFIG = load_experiment().generation
SCHEMA = "TABLE t\ncolumns:\n  A VARCHAR: a column"


# --------------------------------------------------------------- prompts ----


def test_prompts_share_everything_except_the_mode_rule():
    forced = prompts.build_forced(SCHEMA, "  How many rows?  ")
    abstain = prompts.build_abstain(SCHEMA, "How many rows?")

    for messages in (forced, abstain):
        assert [m["role"] for m in messages] == ["system", "user"]
        system = messages[0]["content"]
        assert SCHEMA in system and "DuckDB" in system and "YYYYMMDD" in system
        assert "Use only the tables and columns listed" in system
        assert prompts.OUTPUT_FORMAT in system
        assert system.index("DATABASE SCHEMA") < system.index("RULES")  # schema first
    assert forced[1] == abstain[1] == {"role": "user", "content": "Question: How many rows?"}

    assert prompts.FORCED_RULE in forced[0]["content"]
    assert prompts.ABSTAIN_RULE in abstain[0]["content"]
    without = [
        forced[0]["content"].replace(prompts.FORCED_RULE, ""),
        abstain[0]["content"].replace(prompts.ABSTAIN_RULE, ""),
    ]
    assert without[0] == without[1]
    assert isinstance(prompts.PROMPT_VERSION, str) and prompts.PROMPT_VERSION


# --------------------------------------------------------------- parsing ----


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (GOOD, GOOD),
        (f"<think>let me see {{ }}</think>\n{GOOD}", GOOD),
        (f"<THINK>a</THINK><think>b\nc</think>{GOOD}", GOOD),
        (f"reasoning without an opening tag</think>{GOOD}", GOOD),
        (f"```json\n{GOOD}\n```", GOOD),
        (f"```\n{GOOD}\n```", GOOD),
        (f"<think>x</think>\n```json\n{GOOD}\n```\n", GOOD),
        (f"  \n{GOOD}\n ", GOOD),
    ],
)
def test_clean_output(raw, expected):
    assert clean_output(raw) == expected


def test_parse_output_accepts_answers_and_refusals():
    answer = parse_output(GOOD)
    assert (answer.answerable, answer.sql, answer.confidence, answer.reason) == (
        True, "SELECT 1", 80, "ok",
    )  # fmt: skip
    refusal = parse_output(REFUSAL)
    assert refusal.answerable is False and refusal.sql is None

    extra = parse_output('{"answerable": true, "sql": "SELECT 1", "confidence": 85.0, "x": 1}')
    assert extra.confidence == 85 and extra.reason == ""


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "<think>only thinking</think>",
        "SELECT 1",
        '{"answerable": true, "sql": "SELECT 1", "confidence": 80',  # cut off
        "[1, 2]",
        '{"sql": "SELECT 1", "confidence": 80}',  # answerable missing
        '{"answerable": true, "sql": "SELECT 1", "confidence": 150}',
        '{"answerable": true, "sql": "SELECT 1", "confidence": "high"}',
        '{"answerable": true, "sql": null, "confidence": 80}',  # says yes, gives no query
        '{"answerable": true, "sql": "  ", "confidence": 80}',
        f"Here is the answer: {GOOD}",  # text around the object
    ],
)
def test_parse_output_rejects_malformed_replies(raw):
    with pytest.raises(ParseFailure):
        parse_output(raw)


# ------------------------------------------------------------- generator ----


def test_generate_all_makes_the_seven_calls():
    client = FakeClient()
    generations = generate_all("How many rows?", client, SCHEMA, CONFIG)

    assert [(g.prompt, g.mode, g.temperature, g.seed) for g in generations] == [
        ("forced", "greedy", 0.0, 0),
        ("abstain", "greedy", 0.0, 0),
        ("abstain", "sample", 0.7, 1),
        ("abstain", "sample", 0.7, 2),
        ("abstain", "sample", 0.7, 3),
        ("abstain", "sample", 0.7, 4),
        ("abstain", "sample", 0.7, 5),
    ]
    assert len(client.calls) == 7
    assert [(c["temperature"], c["seed"]) for c in client.calls] == [
        (0.0, 0), (0.0, 0), (0.7, 1), (0.7, 2), (0.7, 3), (0.7, 4), (0.7, 5),
    ]  # fmt: skip
    assert all(c["json_mode"] for c in client.calls)
    assert prompts.FORCED_RULE in client.calls[0]["messages"][0]["content"]
    assert all(prompts.ABSTAIN_RULE in c["messages"][0]["content"] for c in client.calls[1:])

    first = generations[0]
    assert first.parsed == {"answerable": True, "sql": "SELECT 1", "confidence": 80,
                            "reason": "ok"}  # fmt: skip
    assert first.error is None and first.raw == GOOD and len(first.attempts) == 1
    assert (first.latency_ms, first.prompt_tokens, first.completion_tokens) == (100.0, 50, 10)
    json.dumps(first.to_dict())  # must be loggable as JSON


def test_refusal_is_a_valid_generation():
    generation = generate_one(FakeClient([REFUSAL]), [], "abstain", "greedy", 0.0, 0)
    assert generation.error is None
    assert generation.parsed["answerable"] is False and generation.parsed["sql"] is None


def test_malformed_reply_is_retried_once_with_a_corrective_message():
    bad = "<think>hmm</think> Sure! SELECT 1"
    client = FakeClient([bad, f"```json\n{GOOD}\n```"])
    messages = prompts.build_abstain(SCHEMA, "q")
    generation = generate_one(client, messages, "abstain", "sample", 0.7, 3)

    assert generation.error is None and generation.parsed["sql"] == "SELECT 1"
    assert len(client.calls) == 2
    retry = client.calls[1]
    assert retry["messages"][:2] == messages
    assert retry["messages"][2:] == [
        {"role": "assistant", "content": bad},
        {"role": "user", "content": prompts.RETRY_MESSAGE},
    ]
    assert (retry["temperature"], retry["seed"]) == (0.7, 3)
    assert client.calls[0]["messages"] == messages  # the original list was not modified

    assert [a.parse_problem is None for a in generation.attempts] == [False, True]
    assert generation.attempts[0].raw == bad
    # Totals include the retry.
    assert (generation.latency_ms, generation.prompt_tokens, generation.completion_tokens) == (
        200.0, 100, 20,
    )  # fmt: skip


def test_two_malformed_replies_give_parse_error():
    client = FakeClient(["not json", "still not json", GOOD])
    generation = generate_one(client, [], "forced", "greedy", 0.0, 0)
    assert generation.error == PARSE_ERROR and generation.parsed is None
    assert len(client.calls) == 2  # one retry only
    assert generation.raw == "still not json"
    assert all(a.parse_problem for a in generation.attempts)

    no_retry = generate_one(FakeClient(["bad", GOOD]), [], "forced", "greedy", 0.0, 0,
                            max_parse_retries=0)  # fmt: skip
    assert no_retry.error == PARSE_ERROR and len(no_retry.attempts) == 1


def test_second_run_is_served_from_the_cache(tmp_path):
    def reply(call):  # the first reply of every conversation is malformed for seed 2
        is_retry = len(call["messages"]) > 2
        return "garbage" if call["seed"] == 2 and not is_retry else GOOD

    inner = FakeClient(reply)
    client = CachedClient(inner, DiskCache(tmp_path), prompts.PROMPT_VERSION)
    first = generate_all("q", client, SCHEMA, CONFIG)
    # forced and abstain greedy differ in prompt, so 7 calls + 1 retry = 8 real calls.
    assert len(inner.calls) == 8 and client.cache.misses == 8 and client.cache.hits == 0

    second = generate_all("q", client, SCHEMA, CONFIG)
    assert len(inner.calls) == 8 and client.cache.hits == 8
    assert all(a.cached for g in second for a in g.attempts)
    assert [g.parsed for g in second] == [g.parsed for g in first]
    assert [len(g.attempts) for g in second] == [1, 1, 1, 2, 1, 1, 1]

    generate_all("another question", client, SCHEMA, CONFIG)
    assert len(inner.calls) == 16


# ---------------------------------------------------------- smoke script ----


def load_script():
    spec = importlib.util.spec_from_file_location(
        "smoke_generate", ROOT / "scripts" / "smoke_generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_with_fake_clients(tmp_path, monkeypatch, capsys):
    script = load_script()
    inners = []

    def fake_make_client(model, prompt_version, cache_dir=None, env=None):
        inners.append(FakeClient())
        inners[-1].model = model.model  # distinct models must not share cache entries
        return CachedClient(inners[-1], DiskCache(cache_dir), prompt_version)

    monkeypatch.setattr(script, "make_client", fake_make_client)
    out = tmp_path / "smoke.jsonl"
    argv = ["--question", "first throwaway", "--question", "second throwaway",
            "--out", str(out), "--cache-dir", str(tmp_path / "cache"),
            "--catalogue", str(ROOT / "tests" / "fixtures" / "catalogue.yaml")]  # fmt: skip

    assert script.main(argv) == 0
    n_models = len(inners)
    assert n_models >= 1 and all(len(inner.calls) == 14 for inner in inners)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2 * n_models
    assert records[0]["question_id"] == "cli-1" and records[0]["question"] == "first throwaway"
    assert records[0]["prompt_version"] == prompts.PROMPT_VERSION
    assert len(records[0]["generations"]) == 7
    assert "TABLE person" in inners[0].calls[0]["messages"][0]["content"]  # fixture schema used
    printed = capsys.readouterr().out
    assert f"Total calls: {14 * n_models}" in printed and "cache hits" in printed

    # Second run: everything comes from the cache, no new calls.
    assert script.main(argv) == 0
    assert all(len(inner.calls) == 0 for inner in inners[n_models:])
    assert "cached" in capsys.readouterr().out


def test_smoke_script_reads_first_n_from_jsonl_and_handles_empty(tmp_path, monkeypatch, capsys):
    script = load_script()
    monkeypatch.setattr(
        script, "make_client",
        lambda model, version, cache_dir=None, env=None: CachedClient(
            FakeClient(), DiskCache(cache_dir), version
        ),
    )  # fmt: skip
    questions = tmp_path / "questions.jsonl"
    questions.write_text("", encoding="utf-8")
    base = ["--questions", str(questions), "--out", str(tmp_path / "s.jsonl"),
            "--cache-dir", str(tmp_path / "cache"), "--models", "groq",
            "--catalogue", str(ROOT / "tests" / "fixtures" / "catalogue.yaml")]  # fmt: skip
    assert script.main(base) == 1
    assert "No questions" in capsys.readouterr().err

    questions.write_text(
        "".join(json.dumps({"id": f"Q-{i}", "question": f"dummy {i}"}) + "\n" for i in range(5)),
        encoding="utf-8",
    )
    assert script.main([*base, "--n", "2"]) == 0
    records = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert [r["question_id"] for r in records] == ["Q-0", "Q-1"]
    assert {r["model_key"] for r in records} == {"groq"}



# ------------------------------------------------ degenerate output (P6) ----


class LoopsOnSeed3(FakeClient):
    def chat(self, messages, temperature, seed, json_mode=True):
        if seed == 3:
            self.calls.append({"seed": seed})
            raise DegenerateOutput("ollama HTTP 500: token repeat limit reached",
                                   latency_ms=500.0)  # fmt: skip
        return super().chat(messages, temperature, seed, json_mode)


def test_looping_sample_becomes_degenerate_output_without_retry(tmp_path):
    inner = LoopsOnSeed3()
    client = CachedClient(inner, DiskCache(tmp_path), prompts.PROMPT_VERSION)
    generations = generate_all("q", client, SCHEMA, CONFIG)

    looped = generations[4]  # samples start at index 2; seed 3 is the third sample
    assert (looped.seed, looped.error, looped.parsed, looped.raw) == (
        3, DEGENERATE_OUTPUT, None, "",
    )  # fmt: skip
    assert len(looped.attempts) == 1  # no retry
    assert looped.attempts[0].parse_problem.startswith("ollama HTTP 500")
    assert looped.latency_ms == 500.0 and looped.prompt_tokens == 0
    assert [g.error for g in generations].count(DEGENERATE_OUTPUT) == 1
    assert len(inner.calls) == 7  # the question still completes: every call made once

    # A rerun comes entirely from the cache, the loop included.
    again = generate_all("q", client, SCHEMA, CONFIG)
    assert len(inner.calls) == 7
    assert again[4].error == DEGENERATE_OUTPUT and again[4].attempts[0].cached is True
    json.dumps(again[4].to_dict())
