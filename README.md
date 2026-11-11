# ClaimsSQL-Trust

Measures how often small open LLMs hallucinate SQL on healthcare claims data
(CMS DE-SynPUF Sample 1), and whether a verification layer lets them abstain
instead of returning confident wrong answers.

Status: **P3 — benchmark tooling** (questions not written yet).

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

## Command-line tools

| Command | Phase | What it does |
|---|---|---|
| `python scripts/ping_models.py` | P0 | Sends one tiny JSON-mode prompt to every model in `configs/models.yaml` |
| `python -m claimssql.db.load_synpuf` | P1 | Builds `data/claims.duckdb` (4 tables) from `data/raw/`; writes `row_counts.csv` and `load_checks.csv` to `results/tables/`. Safe to re-run. |
| `python -m claimssql.db.profile` | P1 | Writes `column_profile.csv` and `low_cardinality_values.json` to `results/tables/` |
| `python scripts/make_catalogue.py` | P2 | Drafts `schema/catalogue.yaml` from the database, the column profile and the CMS codebook (needs `pip install -e ".[catalogue]"`). Never overwrites an existing catalogue; use `--out` for a fresh draft. |
| `python -m claimssql.schema.catalogue` | P2 | Validates the catalogue against the database; exit code 1 on mismatches |
| `python -m claimssql.schema.context_builder` | P2 | Prints the schema context that goes into the prompt, with an approximate token count |
| `python -m claimssql.eval.validate_benchmark` | P3 | Reads `benchmark/questions.yaml` (never modifies it), checks every record, runs every gold SQL read-only and writes `benchmark/questions.jsonl` with `gold_result_hash` and `split`, plus previews and `benchmark_composition.csv`. `--no-split` while writing questions; `--dry-run` writes nothing. Exit code 1 on invalid records or failing gold SQL. |
| `python scripts/try_sql.py "SELECT ..."` | P3 | Runs one query read-only and prints the first 20 rows, the row count and the time. `-f query.sql` reads the query from a file. |

## Repo layout

Most folders are empty until their phase is built.

```
configs/            models.yaml, experiment.yaml
data/               raw CMS CSVs and DuckDB files (git-ignored)
schema/             catalogue.yaml
docs/               CMS codebook PDF (source of the catalogue descriptions)
benchmark/          questions.yaml (owner-written), questions.jsonl (generated),
                    TEMPLATE.md, trap_tags.yaml, gold_previews/ (git-ignored)
scripts/            ping_models.py, make_catalogue.py, try_sql.py
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
notebooks/          analysis
results/            runs/, tables/, figures/ (cache/ is git-ignored)
tests/
```

## Results

_To be filled from `results/tables/` and `results/figures/` (P6–P7)._
