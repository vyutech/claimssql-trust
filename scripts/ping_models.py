"""Send one tiny JSON-mode prompt to every model in configs/models.yaml.

P0 smoke test: proves Ollama is up, the models are pulled, and the Groq key
works. It talks to the two HTTP APIs directly with httpx; the real, cached
clients arrive in P4 (src/claimssql/llm/).

Usage:
    python scripts/ping_models.py            # ping every model
    python scripts/ping_models.py --only qwen
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from claimssql.config import EnvSettings, ModelConfig, load_env, load_models

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MESSAGES = [
    {"role": "system", "content": "Reply with JSON only."},
    {"role": "user", "content": 'Return exactly this JSON object: {"ok": true}'},
]


def ping_ollama(client: httpx.Client, env: EnvSettings, model: str) -> str:
    r = client.post(
        f"{env.ollama_host}/api/chat",
        json={
            "model": model,
            "messages": MESSAGES,
            "stream": False,
            "format": "json",
            # Ping only checks reachability, so skip the reasoning phase: on CPU a
            # thinking model spends ~90 s reasoning about a one-line reply.
            # Whether the benchmark runs with thinking on or off is decided in P4.
            "think": False,
            "options": {"temperature": 0, "seed": 0},
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ping_groq(client: httpx.Client, env: EnvSettings, model: str) -> str:
    if not env.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is empty - paste it into .env")
    r = client.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {env.groq_api_key}"},
        json={
            "model": model,
            "messages": MESSAGES,
            "temperature": 0,
            "seed": 0,
            "response_format": {"type": "json_object"},
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ping(client: httpx.Client, env: EnvSettings, m: ModelConfig) -> tuple[bool, str, float]:
    """Return (ok, reply-or-error, latency_ms). Never includes the API key."""
    fn = ping_ollama if m.provider == "ollama" else ping_groq
    start = time.perf_counter()
    try:
        text = fn(client, env, m.model)
        ok = bool(text.strip())
    except httpx.HTTPStatusError as e:
        # Response body explains the failure (e.g. model not found); headers are not printed.
        ok, text = False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:  # noqa: BLE001 - a ping should report, not crash
        ok, text = False, f"{type(e).__name__}: {e}"
    return ok, " ".join(text.split())[:80], (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="ping just the model with this key")
    # First call to a local model includes loading it into memory, hence the long default.
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per call")
    args = parser.parse_args()

    env = load_env()
    models = load_models().models
    if args.only:
        models = [m for m in models if m.key == args.only]
        if not models:
            parser.error(f"no model with key {args.only!r} in models.yaml")

    all_ok = True
    with httpx.Client(timeout=args.timeout) as client:
        for m in models:
            ok, text, ms = ping(client, env, m)
            all_ok &= ok
            status = "OK  " if ok else "FAIL"
            print(f"{status} {m.key:<8} {m.provider:<7} {m.model:<28} {ms:>8.0f} ms  {text}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
