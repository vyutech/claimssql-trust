# Deploying the demo to Streamlit Community Cloud

The demo is the Streamlit app in `app/streamlit_app.py`. It needs no full
database: it ships a 5,000-member demo database for live questions and a small
precomputed file for the benchmark explorer. Live questions use the Groq model
marked `use: demo` in `configs/models.yaml`, with your API key as an app secret.

It runs on [Streamlit Community Cloud](https://share.streamlit.io), which is
free and deploys from a GitHub repository. Hugging Face Spaces was the first
plan, but a Streamlit app there now needs the Docker SDK, and Docker Spaces
need a paid plan to create (checked November 2026).

The FastAPI service (`claimssql.api.main`) is not deployed: the hosted app runs
the pipeline itself. Run the API locally instead (see the end of this page).

Websites change their pages from time to time. The labels below are how they
looked when this was written (November 2026); if one differs, look for the
nearest match.

## 1. Build the demo files (laptop)

```powershell
python scripts/build_demo_db.py
python scripts/make_space_bundle.py
```

The first command writes `data/demo.duckdb` and `app_data/` (the explorer data,
the headline results and the example questions). The second writes
`dist/space/`, which is exactly what the hosted app needs. It contains no
`.env`, no full database and no LLM cache. Its `data/` folder holds only the
demo database. (The folder name and the `README.md` header come from the
Hugging Face plan; Community Cloud ignores the header, and GitHub shows it as a
small table.)

Rebuild both after a new benchmark run, so the app shows the new results.
`build_demo_db.py` also copies the numbers and chart data (from `results/tables/`),
so run the report first.

The app has no sidebar until a code link is set. To show a "Code on GitHub"
button, set `GITHUB_URL` at the top of `app/ui_common.py` to the public
repository, then build the bundle. Leave it empty to keep the sidebar hidden.

### Optional: run the bundle locally first

From `dist/space`, the way Community Cloud runs it:

```powershell
cd dist\space
$env:GROQ_API_KEY = "<your key>"
streamlit run app/streamlit_app.py
```

The bundle has no `.env`, so the key must come from the environment, as it will
online. Close the terminal afterwards so the key does not linger. Running the
app creates `__pycache__` folders in `dist/space`; build the bundle again
before you copy it in step 2.

## 2. Put the bundle in its own GitHub repository (once)

`make_space_bundle.py` deletes and recreates `dist/space` on every build, so the
deploy repository lives in a separate folder.

1. On GitHub, **New repository** (<https://github.com/new>).
   **Repository name**: `claimssql-trust-demo`. **Public** (the bundle holds no
   secrets and the data is synthetic). Do **not** add a README, .gitignore or
   license, so the repository starts empty. **Create repository**.
2. Copy the bundle and push it with git. GitHub's web upload accepts files up to
   25 MB only, and `data/demo.duckdb` is about 39 MB, so use git (its limit is
   100 MB per file):

   ```powershell
   Copy-Item -Recurse dist\space ..\claimssql-trust-demo
   cd ..\claimssql-trust-demo
   git init -b main
   git add .
   git commit -m "ClaimsSQL-Trust demo bundle"
   git remote add origin https://github.com/<your-name>/claimssql-trust-demo.git
   git push -u origin main
   ```

3. On GitHub, check that `.streamlit/config.toml` and `data/demo.duckdb` are
   listed. Without the first the app falls back to Streamlit's default look;
   without the second Try it cannot answer.

## 3. Create the app on Community Cloud

1. <https://share.streamlit.io> → **Continue with GitHub**, and allow it to read
   your repositories.
2. **Create app** (top right) → **Yup, I have an app**.
3. **Repository**: `<your-name>/claimssql-trust-demo`. **Branch**: `main`.
   **Main file path**: `app/streamlit_app.py`.
   **App URL** (optional): a subdomain, for example `claimssql-trust`.
4. **Advanced settings**:
   - **Python version**: 3.12 (the default; the project needs 3.11 or newer).
   - **Secrets**: paste one line with your Groq key:

     ```toml
     GROQ_API_KEY = "<your key>"
     ```

     It must be a top-level line, not under a `[section]`: Streamlit copies
     top-level secrets into the environment when the app starts, and the app
     reads the key from the environment, as on the laptop. Secrets are not
     stored in the repository and are not shown to visitors.
   - **Save**.
5. **Deploy**. The first start takes a few minutes while the packages in
   `requirements.txt` install.

To change the key later: the app's **⋮** menu in your workspace → **Settings**
→ **Secrets**.

## 4. Check it

1. If the app shows an error instead of starting, open the logs panel on the
   right (only you see it) and send me the last lines.
2. **Home**, **How it works**, **Results** and **About** need no API key; check
   that the charts on Results show.
3. **Explorer**: pick a model and a question. It needs no API key.
4. **Try it**: click an example question. It takes 1-2 minutes on Groq's free
   tier (8,000 tokens per minute) and should end with **Answered** and a
   number. About 15 questions per day fit in the free daily limit; after that
   the app says so instead of failing.

An app with no visitors for 12 hours goes to sleep. Anyone who opens it then
sees a "Yes, get this app back up!" button; the app restarts in a minute or so.
Open it yourself shortly before showing it to someone (for example the viva).

## 5. Update the app

After a change to the app or new results: build again (step 1), mirror the
bundle into the deploy folder while keeping its `.git`, then push. The app
redeploys by itself.

```powershell
python scripts/build_demo_db.py
python scripts/make_space_bundle.py
robocopy dist\space ..\claimssql-trust-demo /MIR /XD .git
cd ..\claimssql-trust-demo
git add -A
git commit -m "Update demo bundle"
git push
```

`/MIR` makes the deploy folder an exact copy of the bundle, so files removed
from the bundle are removed there too; `/XD .git` keeps the git history.

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
