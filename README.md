# ClaimsSQL-Trust

Measures how often small open LLMs hallucinate SQL on healthcare claims data
(CMS DE-SynPUF Sample 1), and whether a verification layer lets them abstain
instead of returning confident wrong answers.

Status: **P6 — evaluation** (benchmark runner, scoring, metrics, report; full run pending).

## Benchmark

The owner writes 200 questions in `benchmark/questions.yaml` (120 answerable with
gold SQL, 80 infeasible). The validator generates `benchmark/questions.jsonl` from
it, adding each gold result's hash and the dev/test split. `benchmark/TEMPLATE.md`
explains the record format and `benchmark/trap_tags.yaml` lists the allowed trap
tags. Results are compared with `src/claimssql/eval/compare.py`: column names and
row order are ignored, numbers are rounded to 4 decimal places.

## Schema catalogue

`schema/catalogue.yaml` describes every table and column. Descriptions and code
meanings are taken from the official
[CMS Linkable 2008–2010 Medicare DE-SynPUF Codebook](https://www.cms.gov/files/document/de-10-codebook.pdf-0)
(copy in `docs/de-10-codebook.pdf`) and reviewed by the project owner, who also
writes the `notes` fields. Notes are not shown to the models.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env   # then paste your GROQ_API_KEY into .env
pytest
```

Requirements: Python 3.11+, [Ollama](https://ollama.com) running locally, a Groq API key.

## Models

Model names live only in `configs/models.yaml`. Pull the local ones with
`ollama pull <name>` and check everything replies:

```powershell
python scripts/ping_models.py
```

## Data

Download CMS DE-SynPUF Sample 1 into `data/raw/` (beneficiary 2008–2010,
inpatient, outpatient, carrier 1A + 1B; not the prescription drug file).
`data/` is git-ignored. Then build the database and profile it:

```powershell
python -m claimssql.db.load_synpuf
python -m claimssql.db.profile
```

ID and code columns are loaded as text so leading zeros survive; `*_DT` columns
stay as raw `YYYYMMDD` integers.

## Generation

For each question and model the generator makes 7 calls: one with the `forced`
prompt (must write SQL), one with the `abstain` prompt (may refuse), both at
temperature 0, and 5 sampled `abstain` calls at temperature 0.7. Replies are JSON:
`{"answerable", "sql", "confidence", "reason"}`. A reply that cannot be parsed gets
one retry, then counts as `PARSE_ERROR`.

Every call goes through a disk cache in `results/cache/` (git-ignored), keyed by
provider, model, prompt version, messages, temperature, seed and the model's
options in `configs/models.yaml`. Re-running costs no calls. Changing any prompt
text requires bumping `PROMPT_VERSION` in `src/claimssql/generate/prompts.py`.

## Verification and the answer/abstain decision

Three verifiers look at the SQL a model wrote (`src/claimssql/verify/`):

- **V1 schema check** parses the SQL without running it and reports tables and
  columns that are not in the catalogue. Names the query defines itself (CTEs,
  aliases, subquery columns) are not errors.
- **V2 dry run** accepts only a single read-only query, runs it on a read-only
  connection with a timeout, and classifies any error (parse, binder, catalog,
  conversion, timeout, other, forbidden).
- **V3 self-consistency** runs the 5 sampled queries and groups them by result.
  Agreement is the size of the largest group divided by 5.

`src/claimssql/policy/decide.py` turns one logged record into a decision for
each configuration, without calling the model again:

| Config | Answers when |
|---|---|
| C0 | the `forced` prompt produced SQL |
| C1 | the `abstain` prompt produced SQL (the model did not refuse) |
| C2 | C1 and V1 passes |
| C3 | C2 and V2 passes |
| C4 | C3, the greedy result is in the largest V3 group, and agreement ≥ τ |

Otherwise it abstains with one reason code: `MODEL_REFUSED`, `SCHEMA_UNKNOWN`,
`EXEC_ERROR`, `LOW_AGREEMENT`, `PARSE_ERROR` or `DEGENERATE_OUTPUT`
(the model looped and the server aborted the reply; such replies are cached,
since the same call loops again, and counted in `e3_generation_failures.csv`).

## Running the benchmark

The run is split in two stages that share the LLM disk cache, because the models
run on a Kaggle GPU and the database lives on the laptop:

1. **generate** (Kaggle): `python -m claimssql.eval.run_benchmark --stage generate --run-id full`
   asks every benchmark model every question (7 calls each). It needs only
   `benchmark/questions.jsonl` and `schema/catalogue.yaml`.
2. Copy the cache from Kaggle to the laptop (`scripts/merge_cache.py`). Both machines must be on the
   same commit: the cache key includes the prompt text.
3. **verify** (laptop): `python -m claimssql.eval.run_benchmark --stage verify --run-id full`
   makes no model calls. It fails if a cached answer is missing, runs V1–V3
   against the database and writes `results/runs/full.jsonl`.
4. **report** (laptop): `python -m claimssql.eval.report --run-id full`.

The Kaggle side is `notebooks/kaggle_generate.ipynb`; the click-by-click steps are in
[docs/KAGGLE.md](docs/KAGGLE.md).

Scoring puts each question into one of five outcome regions (answerable: correct,
wrong, abstained; infeasible: abstained, answered). The Reliability Score RS_c
gives +1 for a correct answer or a justified abstention, 0 for abstaining on an
answerable question and −c for a wrong answer or an answer to an infeasible
question. τ for C4 is chosen per model on the dev split (highest RS_10); all
other numbers are reported on the test split.

Models marked `use: benchmark` in `configs/models.yaml` take part in the run;
`use: demo` (Groq) is only for the demo and small tests.

## Command-line tools

| Command | Phase | What it does |
|---|---|---|
| `python scripts/ping_models.py` | P0 | Sends one tiny JSON-mode prompt to every model in `configs/models.yaml`. `--only <key>` pings one model; `mid_oss` is not installed on the laptop, so it fails there. |
| `python -m claimssql.db.load_synpuf` | P1 | Builds `data/claims.duckdb` (4 tables) from `data/raw/`; writes `row_counts.csv` and `load_checks.csv` to `results/tables/`. Safe to re-run. |
| `python -m claimssql.db.profile` | P1 | Writes `column_profile.csv` and `low_cardinality_values.json` to `results/tables/` |
| `python scripts/make_catalogue.py` | P2 | Drafts `schema/catalogue.yaml` from the database, the column profile and the CMS codebook (needs `pip install -e ".[catalogue]"`). Never overwrites an existing catalogue; use `--out` for a fresh draft. |
| `python -m claimssql.schema.catalogue` | P2 | Validates the catalogue against the database; exit code 1 on mismatches |
| `python -m claimssql.schema.context_builder` | P2 | Prints the schema context that goes into the prompt, with an approximate token count |
| `python -m claimssql.eval.validate_benchmark` | P3 | Reads `benchmark/questions.yaml` (never modifies it), checks every record, runs every gold SQL read-only and writes `benchmark/questions.jsonl` with `gold_result_hash` and `split`, plus previews and `benchmark_composition.csv`. `--no-split` while writing questions; `--dry-run` writes nothing. Exit code 1 on invalid records or failing gold SQL. |
| `python scripts/try_sql.py "SELECT ..."` | P3 | Runs one query read-only and prints the first 20 rows, the row count and the time. `-f query.sql` reads the query from a file; `--vertical` prints each row as `column: value` lines for wide tables. |
| `python scripts/smoke_generate.py` | P4 | Runs the first 10 questions of `questions.jsonl` on every model (7 calls per question per model), writes `results/runs/smoke.jsonl` and prints cache hits/misses, calls and average seconds per call. `--n`, `--models`, `--question "..."`. Resumable through the disk cache. |
| `python -m claimssql.verify.schema_check` | P5 | Runs the schema check (V1) on every gold query in `questions.jsonl`. All must pass; a failure would be a false positive in V1. Exit code 1 if any fails. |
| `python -m claimssql.eval.run_benchmark --stage generate\|verify\|all` | P6 | Runs the benchmark. `generate` makes the LLM calls (no database needed, resumable through the cache); `verify` re-reads them from the cache, runs V1–V3 and appends to `results/runs/<run-id>.jsonl`. Options: `--run-id`, `--models`, `--groups`, `--split`, `--ids`, `--limit`. Exit code 2 = stopped by a rate limit; run again to continue. |
| `python -m claimssql.eval.report --run-id X` | P6 | Scores a run offline and writes the report tables (`e1`–`e5`, `outcomes_confusion.csv`, `tau_selection.csv`, `failures.jsonl`) to `results/tables/`. |
| `python scripts/make_kaggle_bundle.py` | P6 | Zips what generation needs (`src/`, `configs/`, the catalogue, `questions.jsonl`, `pyproject.toml`) into `dist/claimssql_bundle.zip`, with the git commit and a cache-key fingerprint inside. Warns about uncommitted changes. |
| `python scripts/merge_cache.py cache_qwen.zip ...` | P6 | Copies the LLM cache downloaded from Kaggle into `results/cache/`, skipping files already present. Warns if a zip was made from a different commit. |

## Repo layout

Most folders are empty until their phase is built.

```
configs/            models.yaml, experiment.yaml
data/               raw CMS CSVs and DuckDB files (git-ignored)
schema/             catalogue.yaml
docs/               CMS codebook PDF (source of the catalogue descriptions), KAGGLE.md
benchmark/          questions.yaml (owner-written), questions.jsonl (generated),
                    TEMPLATE.md, trap_tags.yaml, gold_previews/ (git-ignored)
scripts/            ping_models.py, make_catalogue.py, try_sql.py, smoke_generate.py,
                    make_kaggle_bundle.py, merge_cache.py
src/claimssql/
  config.py         loads configs/ and .env
  db/               loading and profiling the claims data
  schema/           codebook parser, catalogue draft/loader/validator, context builder
  llm/              Ollama and Groq clients, disk cache
  generate/         prompts, generator
  verify/           schema check, dry run, self-consistency
  policy/           answer/abstain decision
  eval/             result comparison, benchmark runner, metrics
  finetune/         training-data generation (optional)
  api/              FastAPI service
app/                Streamlit UI
notebooks/          kaggle_generate.ipynb (generation on Kaggle), analysis
results/            runs/, tables/, figures/ (cache/ is git-ignored)
tests/
```

## Results

_To be filled from `results/tables/` and `results/figures/` (P6–P7)._
