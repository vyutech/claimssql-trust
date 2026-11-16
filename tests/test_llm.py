"""P4 client and cache tests. HTTP is mocked: nothing here touches the network."""

import json
import random

import httpx
import pytest

from claimssql.config import EnvSettings, ModelConfig
from claimssql.llm import groq_client as gq
from claimssql.llm.base import DegenerateOutput, LLMError, LLMResponse, RateLimitStop
from claimssql.llm.cache import CachedClient, DiskCache, cache_key
from claimssql.llm.factory import make_client
from claimssql.llm.groq_client import GroqClient, retry_delay
from claimssql.llm.ollama_client import OllamaClient
from fakes import GOOD, FakeClient

MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "Question: q"}]


def mock_http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def groq_ok(text: str = GOOD) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 3000, "completion_tokens": 42},
    })  # fmt: skip


# ---------------------------------------------------------------- Ollama ----


def test_ollama_request_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "message": {"content": GOOD}, "prompt_eval_count": 2800, "eval_count": 35,
        })  # fmt: skip

    client = OllamaClient(
        "some-model", "http://localhost:11434/",
        {"think": False, "num_ctx": 8192, "max_tokens": 600}, http=mock_http(handler),
    )  # fmt: skip
    response = client.chat(MESSAGES, temperature=0.7, seed=3)

    assert seen["url"] == "http://localhost:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "some-model" and body["messages"] == MESSAGES
    assert body["stream"] is False and body["format"] == "json" and body["think"] is False
    assert body["options"] == {"temperature": 0.7, "seed": 3, "num_ctx": 8192, "num_predict": 600}
    assert (response.text, response.prompt_tokens, response.completion_tokens) == (GOOD, 2800, 35)
    assert response.latency_ms >= 0 and response.cached is False


def test_ollama_json_mode_off_and_errors():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(404, json={"error": "model 'x' not found"})

    client = OllamaClient("x", "http://h", http=mock_http(handler))
    with pytest.raises(LLMError, match="404"):
        client.chat(MESSAGES, temperature=0, seed=0, json_mode=False)
    assert "format" not in bodies[0] and "num_ctx" not in bodies[0]["options"]

    with pytest.raises(ValueError, match="num_cxt"):
        OllamaClient("x", "http://h", {"num_cxt": 8192})  # misspelt option


# ------------------------------------------------------------------ Groq ----


def make_groq(handler, sleeps: list[float]) -> GroqClient:
    return GroqClient(
        "some/model", "test-key-123", {"reasoning_effort": "low", "max_tokens": 1500},
        http=mock_http(handler), sleep=sleeps.append, rng=random.Random(0),
    )  # fmt: skip


def test_groq_request_body_and_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return groq_ok()

    response = make_groq(handler, []).chat(MESSAGES, temperature=0, seed=5)
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key-123"
    assert seen["body"] == {
        "model": "some/model", "messages": MESSAGES, "temperature": 0, "seed": 5,
        "response_format": {"type": "json_object"}, "reasoning_effort": "low",
        "max_completion_tokens": 1500,
    }  # fmt: skip
    assert (response.text, response.prompt_tokens, response.completion_tokens) == (GOOD, 3000, 42)


def test_groq_waits_for_retry_after_then_backs_off():
    replies = [
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow"}}),
        httpx.Response(503, text="unavailable"),
        groq_ok(),
    ]
    sleeps: list[float] = []
    response = make_groq(lambda request: replies.pop(0), sleeps).chat(MESSAGES, 0, 0)

    assert response.text == GOOD and replies == []
    assert len(sleeps) == 2
    assert 7 <= sleeps[0] <= 8  # the server's retry-after, plus jitter
    assert 4 <= sleeps[1] <= 5  # second retry: 2 s * 2^1, plus jitter


def test_groq_retries_network_errors():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("no route", request=request)
        return groq_ok()

    sleeps: list[float] = []
    assert make_groq(handler, sleeps).chat(MESSAGES, 0, 0).text == GOOD
    assert len(attempts) == 2 and len(sleeps) == 1


def test_groq_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(gq, "MAX_RETRIES", 3)
    attempts, sleeps = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError, match="HTTP 500"):
        make_groq(handler, sleeps).chat(MESSAGES, 0, 0)
    assert len(attempts) == 4 and len(sleeps) == 3  # no sleep after the last attempt


def test_groq_client_errors_are_not_retried_and_hide_the_key():
    attempts, sleeps = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    with pytest.raises(LLMError) as info:
        make_groq(handler, sleeps).chat(MESSAGES, 0, 0)
    assert len(attempts) == 1 and sleeps == []
    assert "test-key-123" not in str(info.value)

    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient("m", None)


def test_groq_invalid_json_from_the_model_is_returned_as_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {
            "code": "json_validate_failed", "failed_generation": "SELECT 1 -- not json",
        }})  # fmt: skip

    response = make_groq(handler, []).chat(MESSAGES, 0, 0)
    assert response.text == "SELECT 1 -- not json"  # the generator will count PARSE_ERROR


def test_retry_delay():
    rng = random.Random(1)
    assert 30 <= retry_delay(0, "30", rng) <= 31
    assert 2.5 <= retry_delay(0, "2.5", rng) <= 3.5
    assert 2 <= retry_delay(0, None, rng) <= 3
    assert 16 <= retry_delay(3, None, rng) <= 17
    assert 120 <= retry_delay(20, None, rng) <= 121  # capped
    assert 4 <= retry_delay(1, "Wed, 21 Oct 2026 07:28:00 GMT", rng) <= 5  # not seconds


# ----------------------------------------------------------------- cache ----


def key(**changes) -> str:
    base = {"provider": "p", "model": "m", "prompt_version": "v1", "messages": MESSAGES,
            "temperature": 0.0, "seed": 0, "options": {"num_ctx": 8192},
            "json_mode": True}  # fmt: skip
    return cache_key(**{**base, **changes})


def test_cache_key_changes_with_every_input():
    assert key() == key() and len(key()) == 64
    assert key(temperature=0) == key(temperature=0.0)
    changed = [
        key(provider="q"), key(model="n"), key(prompt_version="v2"), key(temperature=0.7),
        key(seed=1), key(options={"num_ctx": 4096}), key(json_mode=False),
        key(messages=[*MESSAGES, {"role": "user", "content": "again"}]),
    ]  # fmt: skip
    assert len({key(), *changed}) == 9


def test_cached_client_hits_misses_and_files(tmp_path):
    inner = FakeClient()
    client = CachedClient(inner, DiskCache(tmp_path), "v1")

    first = client.chat(MESSAGES, temperature=0, seed=0)
    second = client.chat(MESSAGES, temperature=0, seed=0)
    assert len(inner.calls) == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text and second.latency_ms == 100.0  # original latency kept
    assert (client.cache.hits, client.cache.misses) == (1, 1)

    client.chat(MESSAGES, temperature=0.7, seed=1)  # different seed/temperature: new call
    assert len(inner.calls) == 2 and client.cache.misses == 2

    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 2 and all(f.parent.name == f.stem[:2] for f in files)
    entry = json.loads(files[0].read_text(encoding="utf-8"))
    assert entry["request"]["messages"] == MESSAGES and "cached" not in entry["response"]

    # A new process (new cache object) still finds the answers on disk.
    again = CachedClient(FakeClient(["should not be used"]), DiskCache(tmp_path), "v1")
    assert again.chat(MESSAGES, temperature=0, seed=0).cached is True
    # A new prompt version does not.
    v2 = CachedClient(FakeClient(), DiskCache(tmp_path), "v2")
    assert v2.chat(MESSAGES, temperature=0, seed=0).cached is False


def test_failed_calls_are_not_cached_and_corrupt_files_are_misses(tmp_path):
    class Failing(FakeClient):
        def chat(self, *args, **kwargs) -> LLMResponse:
            raise LLMError("down")

    client = CachedClient(Failing(), DiskCache(tmp_path), "v1")
    with pytest.raises(LLMError):
        client.chat(MESSAGES, temperature=0, seed=0)
    assert list(tmp_path.rglob("*.json")) == []

    good = CachedClient(FakeClient(), DiskCache(tmp_path), "v1")
    good.chat(MESSAGES, temperature=0, seed=0)
    (path,) = tmp_path.rglob("*.json")
    path.write_text("{ broken", encoding="utf-8")
    assert good.chat(MESSAGES, temperature=0, seed=0).cached is False  # re-fetched
    assert json.loads(path.read_text(encoding="utf-8"))["response"]["text"] == GOOD


# --------------------------------------------------------------- factory ----


def test_make_client_always_wraps_in_the_cache(tmp_path):
    env = EnvSettings(groq_api_key="k", ollama_host="http://localhost:11434")
    local = make_client(
        ModelConfig(key="a", provider="ollama", model="m1", options={"num_ctx": 8192}),
        "v1", cache_dir=tmp_path, env=env,
    )  # fmt: skip
    remote = make_client(ModelConfig(key="b", provider="groq", model="m2"), "v1", tmp_path, env)
    assert isinstance(local, CachedClient) and isinstance(local.inner, OllamaClient)
    assert isinstance(remote, CachedClient) and isinstance(remote.inner, GroqClient)
    assert local.options == {"num_ctx": 8192} and local.prompt_version == "v1"

    no_key = EnvSettings(groq_api_key=None, ollama_host="http://localhost:11434")
    with pytest.raises(LLMError):
        make_client(ModelConfig(key="b", provider="groq", model="m2"), "v1", tmp_path, no_key)



# ---------------------------------------------- degenerate output (P6) ----

LOOP_BODY = '{"error":"prediction aborted, token repeat limit reached"}'


def test_ollama_token_repeat_limit_is_degenerate_output_not_a_plain_error():
    client = OllamaClient("m", "http://h", http=mock_http(
        lambda request: httpx.Response(500, text=LOOP_BODY)))  # fmt: skip
    with pytest.raises(DegenerateOutput) as info:
        client.chat(MESSAGES, temperature=0.7, seed=3)
    assert "token repeat limit" in info.value.message
    assert info.value.latency_ms >= 0 and info.value.cached is False

    for status, body in [(500, '{"error":"model runner has unexpectedly stopped"}'),
                         (503, LOOP_BODY), (400, LOOP_BODY)]:  # fmt: skip
        other = OllamaClient("m", "http://h", http=mock_http(
            lambda request, s=status, b=body: httpx.Response(s, text=b)))  # fmt: skip
        with pytest.raises(LLMError) as plain:
            other.chat(MESSAGES, temperature=0.7, seed=3)
        assert not isinstance(plain.value, DegenerateOutput)


class Looping(FakeClient):
    def chat(self, messages, temperature, seed, json_mode=True):
        self.calls.append(seed)
        raise DegenerateOutput("ollama HTTP 500: token repeat limit reached", latency_ms=1234.0)


def test_degenerate_output_is_cached_and_replayed(tmp_path):
    inner = Looping()
    client = CachedClient(inner, DiskCache(tmp_path), "v1")
    with pytest.raises(DegenerateOutput) as first:
        client.chat(MESSAGES, temperature=0.7, seed=3)
    assert first.value.cached is False

    with pytest.raises(DegenerateOutput) as second:
        client.chat(MESSAGES, temperature=0.7, seed=3)
    assert second.value.cached is True and second.value.latency_ms == 1234.0
    assert "token repeat limit" in second.value.message
    assert inner.calls == [3]  # the model was asked once, not again
    (path,) = tmp_path.rglob("*.json")
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert "response" not in entry and entry["degenerate"]["latency_ms"] == 1234.0

    # A cache-only client (the verify stage) replays it too, without a model.
    replay = CachedClient(FakeClient(["unused"]), DiskCache(tmp_path), "v1")
    with pytest.raises(DegenerateOutput):
        replay.chat(MESSAGES, temperature=0.7, seed=3)


@pytest.mark.parametrize(
    "error", [LLMError("ollama request failed: ReadTimeout"), RateLimitStop("wait 3600 s")]
)
def test_network_errors_and_rate_limits_are_still_not_cached(tmp_path, error):
    class Failing(FakeClient):
        def chat(self, *args, **kwargs):
            raise error

    with pytest.raises(type(error)):
        CachedClient(Failing(), DiskCache(tmp_path), "v1").chat(MESSAGES, 0, 0)
    assert list(tmp_path.rglob("*.json")) == []


def test_cache_keys_of_normal_calls_did_not_change():
    # Pinned before DEGENERATE_OUTPUT was added: the key of an ordinary call
    # must stay the same, or every cache made so far would be lost.
    key = cache_key("ollama", "qwen3.5:4b", "v1", MESSAGES, 0.7, 3,
                    {"think": False, "num_ctx": 8192, "max_tokens": 600}, True)  # fmt: skip
    assert key == "165f9e1acd02dfbfee23d292ef52dffda541041799d07ea915b6bf265d2a72e0"
