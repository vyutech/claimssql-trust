"""Collect exactly what the deployed demo (Streamlit Community Cloud) needs into
dist/space/. The name, the README header and --sdk are left over from the first
plan, Hugging Face Spaces; Community Cloud ignores the header (see docs/DEPLOY.md).

    app/                     the app: entry point, shared helpers, pages
    .streamlit/config.toml   the app's theme
    src/claimssql/           the package (imported from src/, not installed)
    configs/                 models and experiment settings
    schema/catalogue.yaml    the schema the model is shown
    data/demo.duckdb         the 5,000-member demo database
    app_data/                precomputed explorer data, headline results, sample questions
    requirements.txt         from deploy/requirements.txt
    README.md                short description (with the Hugging Face header)
    Dockerfile               only with --sdk docker

Never included: .env, the full database, results/, the LLM cache, CLAUDE.md.

Usage:
    python scripts/make_space_bundle.py              # sdk: streamlit
    python scripts/make_space_bundle.py --sdk docker
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from claimssql.config import REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "dist" / "space"
STREAMLIT_VERSION = "1.67.0"
COPY = [
    "app",  # streamlit_app.py, ui_common.py and the pages in app_pages/
    ".streamlit/config.toml",  # the theme (never secrets.toml)
    "src/claimssql",
    "configs",
    "schema/catalogue.yaml",
    "data/demo.duckdb",
    "app_data",  # explorer.jsonl, results.json, samples.json, the headline figure
]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")


def commit_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def readme(sdk: str, commit: str) -> str:
    template = (REPO_ROOT / "deploy" / "space_header.md").read_text(encoding="utf-8")
    if sdk == "streamlit":
        lines = f"sdk_version: {STREAMLIT_VERSION}\napp_file: app/streamlit_app.py"
    else:
        lines = "app_port: 7860"
    return (template.replace("{sdk}", sdk).replace("{sdk_lines}", lines)
            .replace("{commit}", commit))  # fmt: skip


def build_space(out: Path, sdk: str, root: Path = REPO_ROOT) -> list[str]:
    """Write the bundle; returns the files in it, relative to `out`."""
    missing = [name for name in COPY if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing: {', '.join(missing)} - run python scripts/build_demo_db.py first"
        )
    if out.exists():
        shutil.rmtree(out)
    for name in COPY:
        source, target = root / name, out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=SKIP)
        else:
            shutil.copy2(source, target)
    shutil.copy2(root / "deploy" / "requirements.txt", out / "requirements.txt")
    if sdk == "docker":
        shutil.copy2(root / "deploy" / "Dockerfile", out / "Dockerfile")
    (out / "README.md").write_text(readme(sdk, commit_hash()), encoding="utf-8")
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdk", choices=["streamlit", "docker"], default="streamlit")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    try:
        files = build_space(args.out, args.sdk)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    size = sum((args.out / f).stat().st_size for f in files) / 1e6
    print(f"Wrote {args.out} ({len(files)} files, {size:.1f} MB, sdk: {args.sdk})")
    for name in ("data/demo.duckdb", "app_data/explorer.jsonl", "app_data/results.json"):
        print(f"  {name}: {(args.out / name).stat().st_size / 1e6:.1f} MB")
    print("Next: follow docs/DEPLOY.md to push it to the demo repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
