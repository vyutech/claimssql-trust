# ClaimsSQL-Trust

Measures how often small open LLMs hallucinate SQL on healthcare claims data
(CMS DE-SynPUF Sample 1), and whether a verification layer lets them abstain
instead of returning confident wrong answers.

Status: **P2 — schema catalogue**.

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

## Repo layout

Most folders are empty until their phase is built.

```
configs/            models.yaml, experiment.yaml
data/               raw CMS CSVs and DuckDB files (git-ignored)
schema/             catalogue.yaml
docs/               CMS codebook PDF (source of the catalogue descriptions)
benchmark/          questions.jsonl, template, datasheet
scripts/            ping_models.py
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
