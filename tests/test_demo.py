"""P7A tests: demo database, explorer data, live pipeline, API and Space bundle.

All on the tiny P1 fixture database with a fake model: no network, no full data.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from claimssql.api import main as api
from claimssql.api import pipeline as pl
from claimssql.config import load_experiment, load_models
from claimssql.db import load_synpuf as ls
from claimssql.db.demo import build_demo_db, chosen_ids
from claimssql.eval.compare import result_hash
from claimssql.eval.explorer import build_explorer_rows, jsonable, read_explorer, write_explorer
from claimssql.llm.base import DegenerateOutput, RateLimitStop
from claimssql.schema.catalogue import load_catalogue
from fakes import REFUSAL, FakeClient

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
GENERATION = load_experiment().generation
COUNT = "SELECT count(*) FROM beneficiary"  # 6 in the fixture
COUNT_2008 = "SELECT count(*) FROM beneficiary WHERE year = 2008"  # 3


def answer(sql: str, confidence: int = 90) -> str:
    return json.dumps({"answerable": True, "sql": sql, "confidence": confidence, "reason": "ok"})


@pytest.fixture(scope="module")
def full_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("p7") / "claims.duckdb"
    ls.build_database(FIXTURES / "raw", path)
    return path


@pytest.fixture()
def con(full_db):
    connection = duckdb.connect(str(full_db), read_only=True)
    yield connection
    connection.close()


@pytest.fixture()
def schema(con) -> dict[str, set[str]]:
    rows = con.execute(
        "SELECT table_name, column_name FROM information_schema.columns"
    ).fetchall()
    out: dict[str, set[str]] = {}
    for table, column in rows:
        out.setdefault(table.lower(), set()).add(column.upper())
    return out


# --------------------------------------------------------------- demo DB ----


def test_demo_db_keeps_whole_beneficiaries(full_db, tmp_path):
    out = tmp_path / "demo.duckdb"
    counts = build_demo_db(full_db, out, n=2)
    demo = duckdb.connect(str(out), read_only=True)
    source = duckdb.connect(str(full_db), read_only=True)
    try:
        ids = chosen_ids(demo)
        assert len(ids) == 2
        quoted = ", ".join(f"'{i}'" for i in ids)
        for table in ls.TABLE_SOURCES:
            # Same columns and types as the full database ...
            assert demo.execute(f"DESCRIBE {table}").fetchall() == \
                source.execute(f"DESCRIBE {table}").fetchall()  # fmt: skip
            # ... and every row of the chosen people, nobody else's.
            expected = source.execute(
                f"SELECT count(*) FROM {table} WHERE DESYNPUF_ID IN ({quoted})"
            ).fetchone()[0]
            assert counts[table] == expected
            assert demo.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == expected
    finally:
        demo.close()
        source.close()


def test_demo_db_is_deterministic_and_replaces_the_old_file(full_db, tmp_path):
    first, second, other = tmp_path / "a.duckdb", tmp_path / "b.duckdb", tmp_path / "c.duckdb"
    build_demo_db(full_db, first, n=2)
    build_demo_db(full_db, second, n=2)
    salts = {}
    for salt in ("s1", "s2", "s3", "s4", "s5"):
        build_demo_db(full_db, other, n=1, salt=salt)  # rebuilt in place each time
        c = duckdb.connect(str(other), read_only=True)
        salts[salt] = tuple(chosen_ids(c))
        c.close()

    def ids(path):
        c = duckdb.connect(str(path), read_only=True)
        try:
            return chosen_ids(c)
        finally:
            c.close()

    assert ids(first) == ids(second)
    assert len(set(salts.values())) > 1  # the salt really changes the sample
    assert not list(tmp_path.glob("*.tmp"))
    assert len(ids(tmp_path / "a.duckdb")) == 2 and build_demo_db(full_db, first, n=99)
    assert len(ids(first)) == 3  # asking for more people than exist keeps everyone


# -------------------------------------------------------------- explorer ----


def record(qid, group, split, forced_sql, abstain_sql, agreement, gold_hash):
    def gen(sql):
        parsed = {"answerable": sql is not None, "sql": sql, "confidence": 90, "reason": "x"}
        return {"parsed": parsed, "error": None, "attempts": [], "latency_ms": 1.0,
                "prompt_tokens": 1, "completion_tokens": 1}  # fmt: skip

    def verified(sql, value):
        if sql is None:
            return None, None
        return ({"passed": True, "unknown_tables": [], "unknown_columns": [],
                 "parse_error": None},
                {"ok": True, "error_class": None, "error": None,
                 "result_hash": result_hash([(value,)])})  # fmt: skip

    answerable = group in ("easy", "medium", "hard")
    forced_v1, forced_v2 = verified(forced_sql, 6 if forced_sql == COUNT else 3)
    v1, v2 = verified(abstain_sql, 6 if abstain_sql == COUNT else 3)
    return {
        "question_id": qid, "question": f"question {qid}", "group": group, "split": split,
        "answerable": answerable, "gold_result_hash": gold_hash, "model_key": "m",
        "model": "fake-1", "forced": gen(forced_sql), "abstain": gen(abstain_sql),
        "samples": [], "forced_v1": forced_v1, "forced_v2": forced_v2, "v1": v1, "v2": v2,
        "v3": {"agreement": agreement, "greedy_in_majority": True, "k": 5,
               "cluster_sizes": [int(agreement * 5)]},
    }  # fmt: skip


def test_explorer_rows(con, tmp_path):
    gold6 = result_hash([(6,)])
    records = [
        record("E-1", "easy", "dev", COUNT, COUNT, 1.0, gold6),  # right everywhere
        record("E-2", "easy", "test", COUNT_2008, COUNT, 0.2, gold6),  # C0 wrong; C4 low agreement
        record("T-1", "T1", "test", COUNT, None, 0.0, None),  # C0 answers a trap; C4 refuses
    ]
    questions = {"E-1": {"gold_sql": COUNT, "notes": ""}, "E-2": {"gold_sql": COUNT},
                 "T-1": {"gold_sql": None, "notes": "no such data"}}  # fmt: skip
    rows = build_explorer_rows(records, questions, con)
    by_id = {r["question_id"]: r for r in rows}

    e1 = by_id["E-1"]
    assert e1["gold"] == {"columns": ["count_star()"], "rows": [[6]], "more_rows": False,
                          "error": None}  # fmt: skip
    assert e1["tau"] == 0.0  # one dev record, always right: the tie rule picks the lowest
    assert e1["configs"]["C0"]["correct"] and e1["configs"]["C4"]["correct"]
    assert e1["configs"]["C4"]["result"]["rows"] == [[6]]
    assert e1["configs"]["C4"]["verification"]["agreement"] == 1.0
    assert e1["configs"]["C0"]["verification"]["agreement"] is None  # C0 has no V3

    e2 = by_id["E-2"]["configs"]
    assert (e2["C0"]["answered"], e2["C0"]["correct"], e2["C0"]["result"]["rows"]) == (
        True, False, [[3]],
    )  # fmt: skip
    # With tau 0 even agreement 0.2 answers; the point is the answer and its SQL are kept.
    assert e2["C4"]["model_sql"] == COUNT

    t1 = by_id["T-1"]
    assert t1["gold"] is None and t1["notes"] == "no such data"
    assert t1["configs"]["C0"]["region"] == "infeasible_answered"
    assert (t1["configs"]["C4"]["answered"], t1["configs"]["C4"]["reason_code"]) == (
        False, "MODEL_REFUSED",
    )  # fmt: skip
    assert t1["configs"]["C4"]["result"] is None

    path = tmp_path / "explorer.jsonl"
    write_explorer(path, rows)
    assert read_explorer(path) == rows
    assert read_explorer(tmp_path / "missing.jsonl") == []


def test_explorer_reports_preview_errors_and_converts_values(con):
    rows = build_explorer_rows([record("E-9", "easy", "test", "SELECT nope", COUNT, 1.0, "h")],
                               {"E-9": {"gold_sql": "SELECT 1"}}, con)  # fmt: skip
    assert "nope" in rows[0]["configs"]["C0"]["result"]["error"]
    import datetime as dt
    from decimal import Decimal

    assert jsonable(Decimal("2.50")) == 2.5 and jsonable(dt.date(2008, 1, 4)) == "2008-01-04"
    assert jsonable(None) is None and jsonable([1, Decimal("1")]) == [1, 1.0]


# -------------------------------------------------------------- pipeline ----


def ask(replies, con, schema, **kwargs):
    client = FakeClient(replies)
    messages: list[str] = []
    result = pl.ask("How many members?", client, con, schema, "SCHEMA", GENERATION,
                    progress=messages.append, **kwargs)  # fmt: skip
    return result, client, messages


def test_pipeline_answers_when_everything_agrees(con, schema):
    result, client, messages = ask(lambda call: answer(COUNT), con, schema)
    assert result.answered and result.reason_code is None
    assert (result.sql, result.columns, result.rows) == (COUNT, ["count_star()"], [[6]])
    assert (result.agreement, result.cluster_sizes, result.k) == (1.0, [3], 3)
    assert len(client.calls) == 4  # 1 greedy + k = 3 samples, no forced call
    assert [c["seed"] for c in client.calls] == [0, 1, 2, 3]
    assert result.calls == 4 and result.cached_calls == 0
    assert messages[0] == "Asking the model for an answer" and "Running the query" in messages
    json.dumps(result.to_dict())


@pytest.mark.parametrize(
    ("sample_sqls", "answered"),
    [([COUNT, COUNT, COUNT_2008], True),  # 2 of 3 agree: exactly tau = 2/3
     ([COUNT, COUNT_2008, COUNT_2008], False),  # 1 of 3
     (["SELECT nope", COUNT, COUNT_2008], False)],  # a failing sample agrees with nobody
)  # fmt: skip
def test_pipeline_needs_two_of_three_samples(con, schema, sample_sqls, answered):
    replies = [answer(COUNT), *(answer(sql) for sql in sample_sqls)]
    result, _, _ = ask(replies, con, schema)
    assert result.answered is answered
    if not answered:
        assert result.reason_code == "LOW_AGREEMENT"
        assert result.reason_text == pl.REASON_TEXT["LOW_AGREEMENT"]
        assert result.rows == [] and result.model_sql == COUNT


def test_pipeline_abstains_for_each_reason(con, schema):
    refused, _, _ = ask(lambda call: REFUSAL, con, schema)
    assert (refused.reason_code, refused.agreement, refused.model_sql) == (
        "MODEL_REFUSED", None, None,
    )  # fmt: skip
    assert refused.reason_text.startswith("The model said")

    unknown, _, _ = ask(lambda call: answer("SELECT AGE FROM beneficiary"), con, schema)
    assert (unknown.reason_code, unknown.unknown_columns) == ("SCHEMA_UNKNOWN", ["AGE"])

    broken, _, _ = ask(lambda call: answer("SELECT CAST(DESYNPUF_ID AS INT) FROM beneficiary"),
                       con, schema)  # fmt: skip
    assert broken.reason_code == "EXEC_ERROR" and "Conversion" in broken.exec_error

    class LoopsOnGreedy(FakeClient):
        def chat(self, messages, temperature, seed, json_mode=True):
            if seed == 0:
                raise DegenerateOutput("ollama HTTP 500: token repeat limit reached")
            return super().chat(messages, temperature, seed, json_mode)

    looped = pl.ask("q", LoopsOnGreedy(lambda call: answer(COUNT)), con, schema, "S", GENERATION)
    assert looped.reason_code == "DEGENERATE_OUTPUT"
    assert set(pl.REASON_TEXT) >= {"MODEL_REFUSED", "SCHEMA_UNKNOWN", "EXEC_ERROR",
                                   "LOW_AGREEMENT", "PARSE_ERROR", "DEGENERATE_OUTPUT"}


def test_pipeline_lets_the_daily_limit_through(con, schema):
    class Limited(FakeClient):
        def chat(self, *args, **kwargs):
            raise RateLimitStop("asked to wait 3600 s")

    with pytest.raises(RateLimitStop):
        pl.ask("q", Limited(), con, schema, "S", GENERATION)


def test_demo_model_is_the_groq_entry():
    model = pl.demo_model()
    assert model.use == "demo" and model.provider == "groq"
    with pytest.raises(ValueError, match="use: demo"):
        pl.demo_model([m for m in load_models().models if m.use == "benchmark"])


# ------------------------------------------------------------------- API ----


@pytest.fixture()
def client_for(con, schema):
    def make(replies):
        service = pl.DemoService(FakeClient(replies), con,
                                 load_catalogue(FIXTURES / "catalogue.yaml"), GENERATION,
                                 db_path=Path("demo.duckdb"))  # fmt: skip
        service.schema = schema  # the fake SQL is about the fixture claims database
        api.app.dependency_overrides[api.get_service] = lambda: service
        return TestClient(api.app)

    yield make
    api.app.dependency_overrides.clear()


def test_api_health_and_schema(client_for):
    web = client_for(lambda call: answer(COUNT))
    health = web.get("/health").json()
    assert health["status"] == "ok" and health["model"] == "fake-1"
    assert health["database"] == "demo.duckdb" and "api_key_set" in health

    schema = web.get("/schema").json()
    assert [t["name"] for t in schema["tables"]] == ["person", "claim"]
    claim = schema["tables"][1]["columns"]
    assert "EMPTY_CD" not in [c["name"] for c in claim]  # include: false is not shown
    assert "TABLE person" in schema["context"]


def test_api_ask(client_for):
    web = client_for(lambda call: answer(COUNT))
    response = web.post("/ask", json={"question": "How many members?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True and body["rows"] == [[6]] and body["sql"] == COUNT

    refused = client_for(lambda call: REFUSAL).post("/ask", json={"question": "Blood pressure?"})
    assert refused.json()["reason_code"] == "MODEL_REFUSED"

    assert web.post("/ask", json={"question": ""}).status_code == 422
    assert web.post("/ask", json={"question": "   "}).status_code == 422
    assert web.post("/ask", json={}).status_code == 422


def test_api_daily_limit_is_429(client_for):
    class Limited(FakeClient):
        def chat(self, *args, **kwargs):
            raise RateLimitStop("tokens per day")

    web = client_for(None)
    api.app.dependency_overrides[api.get_service]().client = Limited()
    response = web.post("/ask", json={"question": "q"})
    assert response.status_code == 429 and "daily limit" in response.json()["detail"]


# ---------------------------------------------------------- Space bundle ----


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_root(tmp_path) -> Path:
    root = tmp_path / "repo"
    for name in ("app/streamlit_app.py", "app/app_pages/results.py", "app/ui_common.py",
                 ".streamlit/config.toml", ".streamlit/secrets.toml",
                 "src/claimssql/__init__.py", "configs/models.yaml",
                 "schema/catalogue.yaml", "data/demo.duckdb", "app_data/explorer.jsonl",
                 "app_data/results.json", "src/claimssql/__pycache__/x.pyc", ".env",
                 "data/claims.duckdb", "results/cache/ab/x.json", "CLAUDE.md"):  # fmt: skip
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("secret" if name == ".env" else "x", encoding="utf-8")
    shutil.copytree(ROOT / "deploy", root / "deploy")
    return root


def test_space_bundle_contents(fake_root, tmp_path):
    bundle = load_script("make_space_bundle")
    out = tmp_path / "space"
    files = bundle.build_space(out, "streamlit", root=fake_root)
    assert files == [
        ".streamlit/config.toml", "README.md", "app/app_pages/results.py",
        "app/streamlit_app.py", "app/ui_common.py", "app_data/explorer.jsonl",
        "app_data/results.json", "configs/models.yaml", "data/demo.duckdb", "requirements.txt",
        "schema/catalogue.yaml", "src/claimssql/__init__.py",
    ]  # fmt: skip
    assert ".streamlit/secrets.toml" not in files  # only the theme file is copied
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and "sdk: streamlit" in readme
    assert "app_file: app/streamlit_app.py" in readme and "{" not in readme

    docker = bundle.build_space(out, "docker", root=fake_root)  # replaces the old bundle
    assert "Dockerfile" in docker
    assert "sdk: docker" in (out / "README.md").read_text(encoding="utf-8")


def test_space_bundle_needs_the_demo_files(fake_root, tmp_path):
    bundle = load_script("make_space_bundle")
    (fake_root / "data" / "demo.duckdb").unlink()
    with pytest.raises(FileNotFoundError, match="build_demo_db"):
        bundle.build_space(tmp_path / "space", "streamlit", root=fake_root)


def test_space_requirements_cover_the_app_imports():
    text = (ROOT / "deploy" / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("duckdb", "sqlglot", "httpx", "pyyaml", "pydantic", "streamlit", "pandas",
                    "python-dotenv"):  # fmt: skip
        assert package in text
    assert "scikit-learn" not in text  # the app must not need it
