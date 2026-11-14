"""Copy the LLM cache from Kaggle's cache_<model>.zip files into results/cache.

Files that already exist are left alone, so merging is safe to repeat and
several models' zips can be merged one after another.

Each zip carries the bundle_info.json of the bundle that produced it. If its
commit or cache-key fingerprint differs from this checkout, the answers would
not be found by the verify stage - the script warns about that.

Usage:
    python scripts/merge_cache.py cache_qwen.zip cache_gemma.zip cache_mid_oss.zip
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from claimssql.config import CACHE_DIR, REPO_ROOT
from claimssql.eval.fingerprint import differences, fingerprint

# Only names of this exact shape are extracted: cache/<2 hex>/<64 hex>.json.
# Anything else in the zip (or a path trying to leave the folder) is ignored.
_ENTRY = re.compile(r"^cache/([0-9a-f]{2})/([0-9a-f]{64})\.json$")


def merge_zip(zip_path: Path, cache_dir: Path) -> dict:
    """Extract new cache entries. Returns counts and the zip's bundle_info (or None)."""
    added = skipped = ignored = 0
    info = None
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name == "bundle_info.json":
                info = json.loads(archive.read(name))
                continue
            match = _ENTRY.match(name)
            if not match or match.group(2)[:2] != match.group(1):
                if not name.endswith("/") and name.startswith("cache/"):
                    ignored += 1
                continue
            target = cache_dir / match.group(1) / f"{match.group(2)}.json"
            if target.exists():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            added += 1
    return {"added": added, "skipped": skipped, "ignored": ignored, "info": info}


def local_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def version_warnings(info: dict | None, commit: str | None, here: dict) -> list[str]:
    if info is None:
        return ["no bundle_info.json in the zip: cannot tell which commit produced it"]
    warnings = []
    if commit and info.get("commit") != commit:
        warnings.append(f"made from commit {str(info.get('commit'))[:10]}, "
                        f"but this checkout is at {commit[:10]}")  # fmt: skip
    if info.get("uncommitted_changes"):
        warnings.append("the bundle was built with uncommitted changes")
    if problems := differences(here, info.get("fingerprint", {})):
        warnings.append("cache keys differ from this checkout, so verify will not find "
                        "these answers: " + "; ".join(problems))  # fmt: skip
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zips", nargs="+", type=Path, help="cache_<model>.zip file(s)")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)

    missing = [str(z) for z in args.zips if not z.is_file()]
    if missing:
        print(f"not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    commit, here = local_commit(), fingerprint()
    warned = False
    total_added = 0
    for zip_path in args.zips:
        result = merge_zip(zip_path, args.cache_dir)
        total_added += result["added"]
        print(f"{zip_path.name}: added {result['added']}, already present {result['skipped']}"
              + (f", ignored {result['ignored']} unexpected file(s)" if result["ignored"] else ""))
        for warning in version_warnings(result["info"], commit, here):
            warned = True
            print(f"  WARNING: {warning}", file=sys.stderr)

    entries = sum(1 for _ in args.cache_dir.rglob("*.json")) if args.cache_dir.exists() else 0
    print(f"\n{total_added} file(s) added; {args.cache_dir} now holds {entries} cached call(s)")
    if warned:
        print("Check the warnings above before running the verify stage.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
