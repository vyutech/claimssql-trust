# Running generation on Kaggle

The benchmark's LLM calls (the **generate** stage) run on a free Kaggle GPU. The
answers are stored in the LLM cache, which you download and merge into the
project on the laptop. **Verify** and **report** then run on the laptop, where the
database is.

```
laptop:  make bundle ──► Kaggle: generate (one run per model) ──► laptop: merge cache ─► verify ─► report
```

Kaggle changes its button labels from time to time. The names below are how the
site looked when this was written; if a label differs, look for the nearest match.

## Before you start (once)

- A Kaggle account that is **phone-verified** (needed for GPU and internet):
  profile picture → **Settings** → **Phone verification**.
- Everything you want in the run is **committed** in git. The bundle records the
  commit, and the laptop must be on the same commit when you verify.

## 1. Build the bundle (laptop)

```powershell
python scripts/make_kaggle_bundle.py
```

This writes `dist/claimssql_bundle.zip` and prints the commit hash. If it prints
a WARNING about uncommitted changes, commit them and run it again.

The zip holds only `src/`, `configs/`, `schema/catalogue.yaml`,
`benchmark/questions.jsonl` and `pyproject.toml`. It never contains `.env`, the
data, the results or `CLAUDE.md`.

## 2. Create the dataset (Kaggle, once per commit)

1. Go to <https://www.kaggle.com/datasets> → **New Dataset**.
2. Drag `dist/claimssql_bundle.zip` into the upload box.
3. Title: `claimssql-bundle`. Leave it **Private**.
4. **Create**. Wait until the page shows the files.

After a new commit: build the bundle again, open the dataset → **New Version** →
upload the new zip → **Create**. In the notebook, check that the input shows the
new version (see step 4).

## 3. Create the notebook (Kaggle, once)

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. **File** → **Import Notebook** → upload `notebooks/kaggle_generate.ipynb` → **Import**.
3. Rename it (top left) to `claimssql-generate`.

## 4. Notebook settings (right-hand panel)

1. **Input** → **Add Input** → filter **Your Work** / **Datasets** → pick
   `claimssql-bundle` → **+**.
2. **Session options** (or **Settings**) → **Accelerator** → **GPU T4 x2**.
3. **Internet** → **On**.
4. **Persistence** → leave as **No persistence**.

## 5. Trial run: 5 questions (do this first)

The notebook has not been run on Kaggle before, so check it on a small run.

1. In the first code cell set:
   ```python
   MODEL_ID = "qwen"
   SPLIT = "all"
   LIMIT = 5
   ```
2. Top right: **Save Version** → **Save & Run All (Commit)** → **Save**.
3. You can close the browser. To watch progress: the circle at the bottom left
   of the Kaggle page (**View Active Events**) → the running version → **Open Logs**.
4. When it has finished, open the notebook's page → **Output** tab. You should see
   `cache_qwen.zip`. The last cell prints calls made, cache hits and seconds per call.

Check three things in the log:

- step 4 shows the model loaded and the GPU memory in use (not `100% CPU`);
- step 5 prints `Cache-key fingerprint matches the bundle.`;
- the last cell says `Status : complete` and `Calls made here : 35` (5 questions × 7 calls).

If the trial fails, copy the error from the log before changing anything.

## 6. Full runs: one per model

For each of `qwen`, `gemma`, `mid_oss`:

1. Open the notebook → **Edit**.
2. Set `MODEL_ID` to the model, `LIMIT = None`, `SPLIT = "all"`.
3. **Save Version** → **Save & Run All (Commit)** → **Save**.
4. When finished: **Output** tab → download `cache_<MODEL_ID>.zip`.

Each full run is 200 questions × 7 calls = 1,400 calls. Kaggle stops a session
after 12 hours and allows about 30 GPU hours per week; the trial run's
"seconds per call" tells you how long a full run will take
(1,400 × seconds per call).

`mid_oss` is `gpt-oss:20b` (13.8 GB). Step 4 of the notebook shows whether it
fits on the two T4 GPUs. If it does not load, or runs on CPU, stop and use the
fallback model named in the project spec.

### If a run stops early

The last cell still saves `cache_<MODEL_ID>.zip` with every finished call. To
continue instead of starting again:

1. Open the notebook → **Edit** → **Add Input** → filter **Notebooks** /
   **Your Work** → pick `claimssql-generate` (its own output) → **+**.
2. **Save Version** → **Save & Run All (Commit)** with the same `MODEL_ID`.

The notebook copies the earlier cache in first, so only the missing calls are made.

## 7. Merge the cache (laptop)

Put the downloaded zips anywhere, then:

```powershell
python scripts/merge_cache.py C:\Users\<you>\Downloads\cache_qwen.zip C:\Users\<you>\Downloads\cache_gemma.zip C:\Users\<you>\Downloads\cache_mid_oss.zip
```

It prints how many files were added and how many were already present. Merging
the same zip twice is harmless.

If it prints a **WARNING** that the commit or the cache keys differ, the laptop
is not on the commit the bundle was built from. Check out that commit (it is
printed in the warning and in the notebook's last cell) before verifying;
otherwise verify will report missing generations.

## 8. Verify and report (laptop)

```powershell
python -m claimssql.eval.run_benchmark --stage verify --run-id full
python -m claimssql.eval.report --run-id full
```

- `verify` makes no model calls. If it stops with "generations missing for
  question …", that model's run is incomplete or its cache was not merged.
- `report` writes the tables to `results/tables/`.

To verify models as they arrive, name them:
`python -m claimssql.eval.run_benchmark --stage verify --run-id full --models qwen`.
Verify skips what is already in `results/runs/full.jsonl`, so you can run it again
after merging the next model.
