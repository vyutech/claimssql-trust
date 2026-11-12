"""P6 runner tests: the generate and verify stages with a fake model (no network)."""

import json
import random
from pathlib import Path

import httpx
import pytest

from claimssql.config import ModelConfig, load_experiment, load_models
from claimssql.db import load_synpuf as ls
from claimssql.eval import run_benchmark as rb
from claimssql.eval.compare import result_hash
from claimssql.eval.score import score_record
from claimssql.llm.base import CacheMiss, LLMError, RateLimitStop
from claimssql.llm.cache import CachedClient, DiskCache
from claimssql.llm.factory import CacheOnlyClient, make_client
from claimssql.llm.groq_client import GroqClient
from claimssql.llm.ollama_client import OllamaClient
from fakes import GOOD, REFUSAL, FakeClient

FIXTURES = Path(__file__).parent / "fixtures"
EXPERIMENT = load_experiment()
COUNT_SQL = "SELECT count(*) FROM beneficiary"  # 6 rows in the fixture database
ANSWER = json.dumps({"answerable": True, "sql": COUNT_SQL, "confidence": 80, "reason": "ok"})
MODEL = ModelConfig(key="fake", provider="ollama", model="fake-1", options={"num_ctx": 8192})

QUESTIONS = [
    {"id": "E-1", "question": "dummy answerable question", "group": "easy", "answerable": True,
     "gold_sql": COUNT_SQL, "gold_result_hash": result_hash([(6,)]), "split": "dev"},
    {"id": "T-1", "question": "dummy infeasible question", "group": "T1", "answerable": False,
     "gold_sql": None, "gold_result_hash": None, "split": "dev"},
    {"id": "M-1", "question": "another dummy question", "group": "medium", "answerable": True,
     "gold_sql": COUNT_SQL, "gold_result_hash": result_hash([(6,)]), "split": "test"},
]  # fmt: skip


def reply(call: dict) -> str:
    """The fake model answers every question except the 'infeasible' one."""
    return REFUSAL if "infeasible" in call["messages"][1]["content"] else ANSWER


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("p6") / "claims.duckdb"
    ls.build_database(FIXTURES / "raw", path)
    return path


@pytest.fixture()
def fake_models(monkeypatch):
    """Route make_client to fake models; cache-only requests stay real."""
    created: list[FakeClient] = []

    def fake_make_client(model, prompt_version, cache_dir=None, env=None, cache_only=False):
        if cache_only:
            return make_client(model, prompt_version, cache_dir=cache_dir, cache_only=True)
        inner = FakeClient(reply)
        inner.provider, inner.model, inner.options = model.provider, model.model, model.options
        created.append(inner)
        return CachedClient(inner, DiskCache(cache_dir), prompt_version)

    monkeypatch.setattr(rb, "make_client", fake_make_client)
    return created


@pytest.fixture()
def questions_file(tmp_path) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS), encoding="utf-8")
    return path


def cli(tmp_path, db_path, questions_file, *args) -> int:
    return rb.main([
        "--questions", str(questions_file), "--cache-dir", str(tmp_path / "cache"),
        "--runs-dir", str(tmp_path / "runs"), "--db-path", str(db_path),
        "--models", "qwen", *args,
    ])  # fmt: skip


# ------------------------------------------------------------- selection ----


def test_select_questions():
    ids = lambda **kw: [q["id"] for q in rb.select_questions(QUESTIONS, **kw)]  # noqa: E731
    assert ids() == ["E-1", "T-1", "M-1"]
    assert ids(split="dev") == ["E-1", "T-1"]
    assert ids(groups=["easy", "medium"]) == ["E-1", "M-1"]
    assert ids(ids=["M-1", "T-1"]) == ["T-1", "M-1"]  # file order is kept
    assert ids(split="dev", limit=1) == ["E-1"]
    assert ids(groups=["hard"]) == []


def test_select_models_defaults_to_benchmark_models():
    models = load_models().models
    default = rb.select_models(models, None)
    assert all(m.use == "benchmark" for m in default)
    assert "groq" not in [m.key for m in default]  # demo only: never run by accident
    assert {"qwen", "gemma", "mid_oss"} <= {m.key for m in default}
    assert [m.key for m in rb.select_models(models, ["groq", "qwen"])] == ["groq", "qwen"]
    with pytest.raises(ValueError, match="nope"):
        rb.select_models(models, ["nope"])


# -------------------------------------------------------------- generate ----


def test_generate_fills_the_cache_and_is_resumable(tmp_path, fake_models, capsys):
    cache = tmp_path / "cache"
    code = rb.stage_generate(QUESTIONS, [MODEL], "SCHEMA", EXPERIMENT, cache)
    assert code == rb.EXIT_OK
    assert len(fake_models[0].calls) == 21  # 3 questions x 7 calls
    assert len(list(cache.rglob("*.json"))) == 21
    out = capsys.readouterr().out
    assert "[fake] 3/3 M-1 | calls 21 (new 21, cached 0)" in out and "ETA 0:00:00" in out

    # Second run: nothing left to do, no model calls.
    assert rb.stage_generate(QUESTIONS, [MODEL], "SCHEMA", EXPERIMENT, cache) == rb.EXIT_OK
    assert len(fake_models[1].calls) == 0
    assert "calls 21 (new 0, cached 21) | all cached" in capsys.readouterr().out


def test_generate_stops_cleanly_on_a_daily_rate_limit(tmp_path, monkeypatch, capsys):
    calls = []

    class LimitedClient(FakeClient):
        def chat(self, messages, temperature, seed, json_mode=True):
            if len(calls) >= 10:
                raise RateLimitStop("asked to wait 3600 s")
            calls.append(1)
            return super().chat(messages, temperature, seed, json_mode)

    monkeypatch.setattr(
        rb, "make_client",
        lambda model, version, cache_dir=None, **kw: CachedClient(
            LimitedClient(), DiskCache(cache_dir), version
        ),
    )  # fmt: skip
    code = rb.stage_generate(QUESTIONS, [MODEL], "SCHEMA", EXPERIMENT, tmp_path / "cache")
    assert code == rb.EXIT_RATE_LIMIT
    out = capsys.readouterr().out
    assert "STOPPED at T-1 (1/3 done)" in out and "Run the same command again" in out
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 10  # finished calls are kept


def test_generate_reports_other_failures_and_carries_on(tmp_path, monkeypatch, capsys):
    class Flaky(FakeClient):
        def chat(self, messages, temperature, seed, json_mode=True):
            if "infeasible" in messages[1]["content"]:
                raise LLMError("ollama HTTP 500")
            return super().chat(messages, temperature, seed, json_mode)

    monkeypatch.setattr(
        rb, "make_client",
        lambda model, version, cache_dir=None, **kw: CachedClient(
            Flaky(), DiskCache(cache_dir), version
        ),
    )  # fmt: skip
    code = rb.stage_generate(QUESTIONS, [MODEL], "SCHEMA", EXPERIMENT, tmp_path / "cache")
    assert code == rb.EXIT_ERRORS
    out = capsys.readouterr().out
    assert "T-1: FAILED - ollama HTTP 500" in out and "3/3 M-1" in out  # went on to M-1


# ---------------------------------------------------- verify and the CLI ----


def test_all_stages_end_to_end(tmp_path, db_path, questions_file, fake_models, capsys):
    assert cli(tmp_path, db_path, questions_file, "--stage", "all", "--run-id", "t") == 0
    records = rb.read_run(tmp_path / "runs" / "t.jsonl")
    assert [(r["question_id"], r["model_key"]) for r in records] == [
        ("E-1", "qwen"), ("T-1", "qwen"), ("M-1", "qwen"),
    ]  # fmt: skip

    easy, infeasible = records[0], records[1]
    assert (easy["group"], easy["split"], easy["answerable"]) == ("easy", "dev", True)
    assert easy["gold_result_hash"] == result_hash([(6,)])
    assert len(easy["samples"]) == 5 and easy["forced"]["parsed"]["sql"] == COUNT_SQL
    assert easy["v1"]["passed"] and easy["v2"]["ok"]
    assert easy["v2"]["result_hash"] == easy["forced_v2"]["result_hash"] == result_hash([(6,)])
    assert easy["v3"]["agreement"] == 1.0 and easy["v3"]["greedy_in_majority"]
    assert easy["v3"]["distinct_sql_executed"] == 1  # five identical samples run once
    # The record is exactly what decide() and the scorer need.
    assert score_record(easy, "C4", 0.6).region == "answerable_correct"

    assert infeasible["v1"] is None and infeasible["v2"] is None  # the model refused
    assert infeasible["v3"]["agreement"] == 0.0
    assert score_record(infeasible, "C0").region == "infeasible_abstained"
    assert score_record(infeasible, "C4", 0.6).reason_code == "MODEL_REFUSED"

    # The verify stage made no model calls of its own: all 21 came from generate.
    assert sum(len(c.calls) for c in fake_models) == 21
    assert "verify: wrote 3 record(s), skipped 0" in capsys.readouterr().out


def test_verify_is_resumable_and_appends(tmp_path, db_path, questions_file, fake_models, capsys):
    base = ("--run-id", "r")
    assert cli(tmp_path, db_path, questions_file, "--stage", "all", "--limit", "1", *base) == 0
    assert cli(tmp_path, db_path, questions_file, "--stage", "generate", *base) == 0
    capsys.readouterr()
    assert cli(tmp_path, db_path, questions_file, "--stage", "verify", *base) == 0
    assert "wrote 2 record(s), skipped 1" in capsys.readouterr().out
    assert len(rb.read_run(tmp_path / "runs" / "r.jsonl")) == 3

    assert cli(tmp_path, db_path, questions_file, "--stage", "verify", *base) == 0
    assert "wrote 0 record(s), skipped 3" in capsys.readouterr().out
    assert len(rb.read_run(tmp_path / "runs" / "r.jsonl")) == 3


def test_verify_fails_loudly_when_generations_are_missing(
    tmp_path, db_path, questions_file, fake_models, capsys
):
    # generate only the first question, then ask verify for all three.
    assert cli(tmp_path, db_path, questions_file, "--stage", "generate", "--limit", "1") == 0
    code = cli(tmp_path, db_path, questions_file, "--stage", "verify", "--run-id", "m")
    assert code == rb.EXIT_ERRORS
    err = capsys.readouterr().err
    assert "generations missing for question T-1 on model 'qwen'" in err
    assert "--stage generate" in err
    # The one complete question was written; nothing was invented for the others.
    assert [r["question_id"] for r in rb.read_run(tmp_path / "runs" / "m.jsonl")] == ["E-1"]
    assert sum(len(c.calls) for c in fake_models) == 7  # verify called no model


def test_nothing_selected(tmp_path, db_path, questions_file, capsys):
    code = cli(tmp_path, db_path, questions_file, "--stage", "verify", "--groups", "hard")
    assert code == rb.EXIT_ERRORS and "Nothing to do" in capsys.readouterr().err


# --------------------------------------------------- client changes (P6) ----


def test_cache_only_client_never_calls_a_model(tmp_path):
    client = make_client(MODEL, "v1", cache_dir=tmp_path, cache_only=True)
    assert isinstance(client.inner, CacheOnlyClient)
    assert (client.provider, client.model, client.options) == (
        "ollama", "fake-1", {"num_ctx": 8192},
    )  # fmt: skip
    messages = [{"role": "user", "content": "q"}]
    with pytest.raises(CacheMiss, match="no cached answer for fake-1"):
        client.chat(messages, temperature=0, seed=0)

    # An answer written by a real (here: fake) client with the same identity is found.
    writer = FakeClient([GOOD])
    writer.provider, writer.model, writer.options = "ollama", "fake-1", {"num_ctx": 8192}
    CachedClient(writer, DiskCache(tmp_path), "v1").chat(messages, temperature=0, seed=0)
    assert client.chat(messages, temperature=0, seed=0).cached is True
    # No API key is needed for cache-only work, even for a Groq model.
    groq = ModelConfig(key="g", provider="groq", model="x")
    assert make_client(groq, "v1", cache_dir=tmp_path, cache_only=True).provider == "groq"


def test_groq_long_wait_raises_rate_limit_stop_without_sleeping():
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "3600"},
                              json={"error": {"message": "tokens per day limit"}})  # fmt: skip

    client = GroqClient("m", "key", http=httpx.Client(transport=httpx.MockTransport(handler)),
                        sleep=sleeps.append, rng=random.Random(0))  # fmt: skip
    with pytest.raises(RateLimitStop, match="3600"):
        client.chat([{"role": "user", "content": "q"}], 0, 0)
    assert sleeps == []


def test_ollama_think_level_is_passed_through():
    low = OllamaClient("gpt", "http://h", {"think": "low", "num_ctx": 8192, "max_tokens": 1500})
    assert low.build_body([], 0, 0, True)["think"] == "low"
    off = OllamaClient("qwen", "http://h", {"think": False})
    assert off.build_body([], 0, 0, True)["think"] is False
    assert OllamaClient("x", "http://h").build_body([], 0, 0, True)["think"] is False

    mid = load_models().get("mid_oss")
    assert (mid.provider, mid.use, mid.options["think"]) == ("ollama", "benchmark", "low")
    assert load_models().get("groq").use == "demo"
