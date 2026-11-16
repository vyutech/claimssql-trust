# Deploying the demo to Hugging Face Spaces

The demo is the Streamlit app in `app/streamlit_app.py`. It needs no full
database: it ships a 5,000-member demo database for live questions and a small
precomputed file for the benchmark explorer. Live questions use the Groq model
marked `use: demo` in `configs/models.yaml`, with your API key as a Space secret.

The FastAPI service (`claimssql.api.main`) is not deployed: a Space runs one
app. Run it locally instead (see the end of this page).

Hugging Face changes its pages from time to time. The labels below are how the
site looked when this was written; if one differs, look for the nearest match.

## 0. Check which SDK Hugging Face offers (once)

Hugging Face's documentation is inconsistent about Streamlit Spaces. Open
<https://huggingface.co/new-space> and look at the **Space SDK** choices:

- **Streamlit** listed and free hardware available → use `--sdk streamlit` below.
- Only **Docker** (with a Streamlit template) → use `--sdk docker` below.
- Creating a Space requires a paid plan → stop here. The free alternative is
  Streamlit Community Cloud, which deploys from a GitHub repository; tell me and
  I will write those steps instead.

## 1. Build the demo files (laptop)

```powershell
python scripts/build_demo_db.py
python scripts/make_space_bundle.py --sdk streamlit
```

Use `--sdk docker` in the second command if step 0 said so. This writes
`dist/space/`, which is exactly the Space's content. It contains no `.env`, no
full database and no LLM cache.

Rebuild both after a new benchmark run (for example when `mid_oss` is verified),
so the explorer shows the new model.

## 2. Create the Space

1. <https://huggingface.co/new-space>.
2. **Owner**: you. **Space name**: `claimssql-trust`.
3. **License**: leave empty or choose one.
4. **Space SDK**: the one from step 0. For Docker, pick the **Blank** template.
5. **Space hardware**: **CPU basic (free)**.
6. **Public** or **Private**, as you prefer. **Create Space**.

## 3. Add the API key as a secret

1. In the Space: **Settings** → **Variables and secrets** → **New secret**.
2. **Name**: `GROQ_API_KEY`. **Value**: your Groq key. **Save**.

Use **secret**, not variable: variables are public, secrets are not shown again.
The app reads it from the environment, as on the laptop.

## 4. Upload the files

The easiest way is the web upload:

1. In the Space: **Files** → **+ Contribute** → **Upload files**.
2. Drag the **contents** of `dist/space/` (not the folder itself) into the box.
   Keep the folder structure: `app/`, `src/`, `configs/`, `schema/`, `data/`,
   `app_data/`, plus `README.md` and `requirements.txt` (and `Dockerfile`).
3. **Commit changes to main**.

`data/demo.duckdb` is larger than 10 MB; Hugging Face stores such files with
its large-file storage automatically on upload. If the upload page refuses it,
upload it on its own with the Hugging Face command-line tool instead:

```powershell
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <your-name>/claimssql-trust dist/space . --repo-type space
```

## 5. Check it

1. The Space shows **Building**, then **Running** (a few minutes the first time).
   If it shows **Runtime error**, open **Logs** and send me the last lines.
2. **Benchmark explorer**: pick a model and a question. It needs no API key.
3. **Ask your own question**: try "How many members had diabetes in 2009?".
   It takes 1-2 minutes on Groq's free tier (8,000 tokens per minute) and
   should end with **Answered** and a number. About 15 questions per day fit in
   the free daily limit; after that the app says so instead of failing.

If the Space reports that `sdk_version` is not supported, edit `README.md` in
the Space and change `sdk_version` to the version it suggests.

## Running the API locally

```powershell
uvicorn claimssql.api.main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for the interactive API page:

- `GET /health`: database, model and whether the API key is set
- `GET /schema`: the tables and columns the model is shown
- `POST /ask` with `{"question": "..."}`: an answer or an abstention;
  HTTP 429 when the daily limit is reached

It uses `data/demo.duckdb` unless `CLAIMSSQL_DB` points elsewhere (for example
`data/claims.duckdb` for the full data).
