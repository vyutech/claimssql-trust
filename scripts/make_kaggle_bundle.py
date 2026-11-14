"""Pack what the generate stage needs into dist/claimssql_bundle.zip for Kaggle.

Included (an allow-list, nothing else can get in):
    src/  configs/  schema/catalogue.yaml  benchmark/questions.jsonl  pyproject.toml
plus two generated files:
    bundle_info.json  the git commit, the time, and the cache-key fingerprint
    README.md         a one-line stub (pip needs the file pyproject.toml names)

Never included: .env, data/, results/, CLAUDE.md, tests, the database.

Usage:
    python scripts/make_kaggle_bundle.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from claimssql.config import REPO_ROOT
from claimssql.eval.fingerprint import fingerprint

INCLUDE_DIRS = ("src", "configs")
INCLUDE_FILES = ("schema/catalogue.yaml", "benchmark/questions.jsonl", "pyproject.toml")
SKIP_DIR_NAMES = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
DEFAULT_OUT = REPO_ROOT / "dist" / "claimssql_bundle.zip"


def bundle_files(root: Path) -> list[Path]:
    """Every file that goes into the bundle, relative to `root`, sorted."""
    files = [Path(name) for name in INCLUDE_FILES]
    for directory in INCLUDE_DIRS:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            skipped = (
                not path.is_file()
                or path.suffix in SKIP_SUFFIXES
                or any(part in SKIP_DIR_NAMES or part.endswith(".egg-info")
                       for part in relative.parts)
            )  # fmt: skip
            if not skipped:
                files.append(relative)
    missing = [str(f) for f in files if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"missing from the project: {', '.join(missing)}")
    return sorted(files)


def git_state(root: Path) -> tuple[str, list[str]]:
    """(commit hash, bundled paths with uncommitted changes)."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              check=True).stdout  # fmt: skip

    commit = git("rev-parse", "HEAD").strip()
    changed = git("status", "--porcelain", "--", *INCLUDE_DIRS, *INCLUDE_FILES)
    return commit, [line[3:] for line in changed.splitlines() if line.strip()]


def write_bundle(root: Path, out: Path, info: dict) -> list[str]:
    """Write the zip; returns the names inside it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stub = (f"ClaimsSQL-Trust generation bundle, commit {info['commit']}.\n"
            "Made by scripts/make_kaggle_bundle.py; see docs/KAGGLE.md.\n")  # fmt: skip
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in bundle_files(root):
            archive.write(root / relative, relative.as_posix())
        archive.writestr("bundle_info.json", json.dumps(info, indent=1))
        archive.writestr("README.md", stub)
        return archive.namelist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    commit, dirty = git_state(REPO_ROOT)
    info = {
        "commit": commit,
        "uncommitted_changes": dirty,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "fingerprint": fingerprint(),
    }
    names = write_bundle(REPO_ROOT, args.out, info)

    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {len(names)} files)")
    print(f"Commit: {commit}")
    print(f"Questions: {info['fingerprint']['n_questions']}, "
          f"prompt {info['fingerprint']['prompt_version']}, "
          f"models: {', '.join(info['fingerprint']['probe_keys'])}")
    if dirty:
        print("\nWARNING: these bundled files have uncommitted changes, so the commit hash "
              "does not describe this bundle:", file=sys.stderr)
        for path in dirty:
            print(f"  {path}", file=sys.stderr)
        print("Commit them and build the bundle again before a real run.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
