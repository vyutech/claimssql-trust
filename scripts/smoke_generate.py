"""Smoke test for generation: a few questions on every model, end to end.

Takes the first N questions from benchmark/questions.jsonl (default 10), or the
questions given with --question, and makes the 7 calls per question per model.
Writes one JSON line per question x model and prints call statistics.

Resumable: every call goes through the disk cache, so running it again after an
interruption only makes the calls that are still missing.

Usage:
    python scripts/smoke_generate.py
    python scripts/smoke_generate.py --n 3 --models groq
    python scripts/smoke_generate.py --question "..." --question "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from claimssql.config import CACHE_DIR, REPO_ROOT, RESULTS_RUNS_DIR, load_experiment, load_models
from claimssql.generate.generator import generate_all
from claimssql.generate.prompts import PROMPT_VERSION
from claimssql.llm.base import LLMError
from claimssql.llm.factory import make_client
from claimssql.schema.catalogue import DEFAULT_CATALOGUE_PATH, load_catalogue
from claimssql.schema.context_builder import render_context

DEFAULT_QUESTIONS = REPO_ROOT / "benchmark" / "questions.jsonl"


def load_questions(path: Path, n: int) -> list[tuple[str, str]]:
    """(id, question) for the first n records of questions.jsonl."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out.append((record["id"], record["question"]))
        if len(out) == n:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10, help="questions to take from the file")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--question", action="append", default=[],
                        help="ask this question instead of reading the file (repeatable)")
    parser.add_argument("--models", nargs="+", help="model keys to run (default: all)")
    parser.add_argument("--out", type=Path, default=RESULTS_RUNS_DIR / "smoke.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.question:
        questions = [(f"cli-{i}", text) for i, text in enumerate(args.question, start=1)]
    else:
        questions = load_questions(args.questions, args.n)
    if not questions:
        print(f"No questions: {args.questions} is empty. Run the validator first, or pass "
              "--question.", file=sys.stderr)
        return 1

    models = load_models().models
    if args.models:
        unknown = sorted(set(args.models) - {m.key for m in models})
        if unknown:
            parser.error(f"not in models.yaml: {', '.join(unknown)}")
        models = [m for m in models if m.key in args.models]

    config = load_experiment().generation
    context = render_context(load_catalogue(args.catalogue))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    failed = False

    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        # Model by model, so a local server loads each model once instead of
        # swapping models in and out of memory for every question.
        for model in models:
            client = make_client(model, PROMPT_VERSION, cache_dir=args.cache_dir)
            calls = live_calls = parse_errors = 0
            live_seconds = 0.0
            for question_id, question in questions:
                started = time.perf_counter()
                record = {
                    "run_id": "smoke",
                    "question_id": question_id,
                    "question": question,
                    "model_key": model.key,
                    "provider": model.provider,
                    "model": model.model,
                    "model_options": model.options,
                    "prompt_version": PROMPT_VERSION,
                }
                try:
                    generations = generate_all(question, client, context, config)
                except LLMError as exc:
                    failed = True
                    record["error"] = str(exc)
                    print(f"  {model.key} {question_id}: FAILED - {exc}", flush=True)
                else:
                    record["generations"] = [g.to_dict() for g in generations]
                    for generation in generations:
                        parse_errors += generation.error is not None
                        for attempt in generation.attempts:
                            calls += 1
                            if not attempt.cached:
                                live_calls += 1
                                live_seconds += attempt.latency_ms / 1000
                    wall = time.perf_counter() - started
                    print(f"  {model.key} {question_id}: {len(generations)} generations "
                          f"in {wall:.0f} s", flush=True)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            summary.append((model.key, calls, client.cache.hits, client.cache.misses,
                            live_seconds / live_calls if live_calls else None, parse_errors))

    print(f"\n{'model':<8}{'calls':>7}{'cache hits':>12}{'misses':>8}{'avg s/call':>12}"
          f"{'parse errors':>14}")
    for key, calls, hits, misses, average, parse_errors in summary:
        avg = f"{average:.1f}" if average is not None else "cached"
        print(f"{key:<8}{calls:>7}{hits:>12}{misses:>8}{avg:>12}{parse_errors:>14}")
    print(f"\nTotal calls: {sum(row[1] for row in summary)} "
          f"({len(questions)} question(s) x {len(models)} model(s))")
    print(f"Wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
