"""P6b tests: the Kaggle bundle, cache merging, the key fingerprint and the notebook."""

import ast
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from claimssql.eval import fingerprint as fp

ROOT = Path(__file__).parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bundle = load_script("make_kaggle_bundle")
merge = load_script("merge_cache")

KEY_A = "ab" + "0" * 62
KEY_B = "cd" + "1" * 62


# ----------------------------------------------------------- fingerprint ----


def test_fingerprint_is_stable_and_covers_the_benchmark_models():
    first, second = fp.fingerprint(), fp.fingerprint()
    assert first == second
    assert set(first["probe_keys"]) == {"qwen", "gemma", "mid_oss"}  # not the demo model
    assert all(len(key) == 64 for key in first["probe_keys"].values())
    assert first["n_questions"] == 200 and first["prompt_version"]
    assert fp.differences(first, second) == []


def test_fingerprint_changes_with_the_inputs(tmp_path):
    base = fp.fingerprint()
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": "X", "question": "a different question"}) + "\n",
                         encoding="utf-8")  # fmt: skip
    other = fp.fingerprint(questions_path=questions)
    assert other["questions_sha256"] != base["questions_sha256"]
    assert other["context_sha256"] == base["context_sha256"]
    problems = fp.differences(other, base)
    assert any("questions_sha256" in p for p in problems)
    assert any("cache key for model 'qwen'" in p for p in problems)

    small = fp.fingerprint(catalogue_path=ROOT / "tests" / "fixtures" / "catalogue.yaml")
    assert small["context_sha256"] != base["context_sha256"]
    assert small["probe_keys"]["qwen"] != base["probe_keys"]["qwen"]


def test_fingerprint_ignores_line_endings_of_the_catalogue(tmp_path):
    source = (ROOT / "schema" / "catalogue.yaml").read_bytes().replace(b"\r\n", b"\n")
    unix, windows = tmp_path / "unix.yaml", tmp_path / "windows.yaml"
    unix.write_bytes(source)
    windows.write_bytes(source.replace(b"\n", b"\r\n"))
    assert fp.fingerprint(catalogue_path=unix) == fp.fingerprint(catalogue_path=windows)


# ---------------------------------------------------------------- bundle ----


@pytest.fixture()
def project(tmp_path) -> Path:
    """A small fake project with things that must and must not be bundled."""
    files = {
        "src/claimssql/__init__.py": "",
        "src/claimssql/config.py": "X = 1\n",
        "src/claimssql/__pycache__/config.cpython-314.pyc": "binary",
        "src/claimssql.egg-info/PKG-INFO": "meta",
        "configs/models.yaml": "models: []\n",
        "schema/catalogue.yaml": "tables: []\n",
        "benchmark/questions.jsonl": "{}\n",
        "benchmark/questions.yaml": "- owner file\n",
        "pyproject.toml": "[project]\n",
        ".env": "GROQ_API_KEY=secret\n",
        "CLAUDE.md": "spec\n",
        "data/claims.duckdb": "database",
        "results/cache/ab/x.json": "{}",
        "tests/test_x.py": "",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def test_bundle_contains_only_the_allow_list(project, tmp_path):
    out = tmp_path / "dist" / "bundle.zip"
    info = {"commit": "abc123", "uncommitted_changes": [], "fingerprint": {}}
    names = bundle.write_bundle(project, out, info)
    assert sorted(names) == [
        "README.md", "benchmark/questions.jsonl", "bundle_info.json", "configs/models.yaml",
        "pyproject.toml", "schema/catalogue.yaml", "src/claimssql/__init__.py",
        "src/claimssql/config.py",
    ]  # fmt: skip
    with zipfile.ZipFile(out) as archive:
        assert json.loads(archive.read("bundle_info.json"))["commit"] == "abc123"
        assert "abc123" in archive.read("README.md").decode()
        everything = b"".join(archive.read(name) for name in archive.namelist())
    assert b"secret" not in everything and b"database" not in everything


def test_bundle_fails_if_a_required_file_is_missing(project):
    (project / "benchmark" / "questions.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="questions.jsonl"):
        bundle.bundle_files(project)


def test_real_bundle_file_list_has_no_secrets_or_data():
    names = [f.as_posix() for f in bundle.bundle_files(ROOT)]
    assert "src/claimssql/eval/run_benchmark.py" in names and "configs/models.yaml" in names
    assert "schema/catalogue.yaml" in names and "benchmark/questions.jsonl" in names
    banned = (".env", "CLAUDE.md", "data/", "results/", "tests/", "__pycache__", ".duckdb")
    assert not [n for n in names if any(b in n for b in banned)]
    assert all(n.split("/")[0] in {"src", "configs", "schema", "benchmark", "pyproject.toml"}
               for n in names)  # fmt: skip


def test_bundle_cli_writes_commit_and_warns_about_uncommitted_changes(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(bundle, "git_state", lambda root: ("f" * 40, ["src/claimssql/x.py"]))
    out = tmp_path / "bundle.zip"
    assert bundle.main(["--out", str(out)]) == 0
    with zipfile.ZipFile(out) as archive:
        info = json.loads(archive.read("bundle_info.json"))
    assert info["commit"] == "f" * 40
    assert info["uncommitted_changes"] == ["src/claimssql/x.py"]
    assert info["fingerprint"] == fp.fingerprint()
    captured = capsys.readouterr()
    assert "f" * 40 in captured.out
    assert "WARNING" in captured.err and "src/claimssql/x.py" in captured.err

    monkeypatch.setattr(bundle, "git_state", lambda root: ("a" * 40, []))
    bundle.main(["--out", str(out)])
    assert "WARNING" not in capsys.readouterr().err


def test_git_state_reads_the_real_repository():
    commit, dirty = bundle.git_state(ROOT)
    assert len(commit) == 40 and isinstance(dirty, list)


# ----------------------------------------------------------------- merge ----


def make_cache_zip(path: Path, entries: dict[str, str], info: dict | None) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
        if info is not None:
            archive.writestr("bundle_info.json", json.dumps(info))
    return path


def test_merge_adds_new_files_and_skips_existing_ones(tmp_path):
    cache = tmp_path / "cache"
    (cache / "ab").mkdir(parents=True)
    (cache / "ab" / f"{KEY_A}.json").write_text("LOCAL", encoding="utf-8")
    zip_path = make_cache_zip(
        tmp_path / "cache_qwen.zip",
        {f"cache/ab/{KEY_A}.json": "FROM KAGGLE", f"cache/cd/{KEY_B}.json": "NEW",
         "generate_summary.json": "{}"},
        {"commit": "abc"},
    )  # fmt: skip
    result = merge.merge_zip(zip_path, cache)
    assert (result["added"], result["skipped"], result["ignored"]) == (1, 1, 0)
    assert result["info"] == {"commit": "abc"}
    assert (cache / "ab" / f"{KEY_A}.json").read_text(encoding="utf-8") == "LOCAL"  # kept
    assert (cache / "cd" / f"{KEY_B}.json").read_text(encoding="utf-8") == "NEW"
    # Merging again changes nothing.
    assert merge.merge_zip(zip_path, cache)["added"] == 0


def test_merge_ignores_anything_that_is_not_a_cache_entry(tmp_path):
    cache = tmp_path / "cache"
    zip_path = make_cache_zip(
        tmp_path / "bad.zip",
        {
            "cache/../../evil.json": "x",          # tries to leave the folder
            "cache/zz/not-a-key.json": "x",        # not a sha256 name
            f"cache/ab/{KEY_B}.json": "x",         # folder does not match the key
            f"cache/cd/{KEY_B}.txt": "x",
            f"cache/cd/{KEY_B}.json": "OK",
        },
        None,
    )  # fmt: skip
    result = merge.merge_zip(zip_path, cache)
    assert (result["added"], result["ignored"], result["info"]) == (1, 4, None)
    assert sorted(p.name for p in tmp_path.rglob("*.json")) == [f"{KEY_B}.json"]
    assert not (tmp_path.parent / "evil.json").exists()


def test_version_warnings():
    here = fp.fingerprint()
    good = {"commit": "c" * 40, "uncommitted_changes": [], "fingerprint": here}
    assert merge.version_warnings(good, "c" * 40, here) == []

    assert "no bundle_info.json" in merge.version_warnings(None, "c" * 40, here)[0]
    other_commit = merge.version_warnings(good, "d" * 40, here)
    assert len(other_commit) == 1 and "cccccccccc" in other_commit[0]
    dirty = merge.version_warnings({**good, "uncommitted_changes": ["x"]}, "c" * 40, here)
    assert "uncommitted changes" in dirty[0]
    changed = {**here, "probe_keys": {**here["probe_keys"], "qwen": "0" * 64}}
    keys = merge.version_warnings({**good, "fingerprint": changed}, "c" * 40, here)
    assert "cache keys differ" in keys[0] and "qwen" in keys[0]


def test_merge_cli(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(merge, "local_commit", lambda: "c" * 40)
    here = fp.fingerprint()
    cache = tmp_path / "cache"
    ok = make_cache_zip(tmp_path / "cache_qwen.zip", {f"cache/ab/{KEY_A}.json": "{}"},
                        {"commit": "c" * 40, "fingerprint": here})  # fmt: skip
    old = make_cache_zip(tmp_path / "cache_gemma.zip", {f"cache/cd/{KEY_B}.json": "{}"},
                         {"commit": "d" * 40, "fingerprint": here})  # fmt: skip
    assert merge.main([str(ok), str(old), "--cache-dir", str(cache)]) == 0
    captured = capsys.readouterr()
    assert "cache_qwen.zip: added 1, already present 0" in captured.out
    assert "2 file(s) added" in captured.out and "now holds 2 cached call(s)" in captured.out
    assert captured.err.count("WARNING") == 1 and "dddddddddd" in captured.err

    assert merge.main([str(tmp_path / "missing.zip"), "--cache-dir", str(cache)]) == 1
    assert "not found" in capsys.readouterr().err


# -------------------------------------------------------------- notebook ----


def test_notebook_is_valid_and_every_code_cell_compiles():
    notebook = json.loads((ROOT / "notebooks" / "kaggle_generate.ipynb").read_text("utf-8"))
    assert notebook["nbformat"] == 4
    code = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 8
    for source in code:
        text = source if isinstance(source, str) else "".join(source)
        ast.parse(text)
        assert not any(line.lstrip().startswith(("!", "%")) for line in text.splitlines())

    parameters = code[0] if isinstance(code[0], str) else "".join(code[0])
    for name in ("MODEL_ID", "SPLIT", "LIMIT"):
        assert name in parameters
    everything = "\n".join(c if isinstance(c, str) else "".join(c) for c in code)
    assert '"--stage", "generate"' in everything
    assert "qwen3.5" not in everything and "gemma4" not in everything  # names stay in models.yaml
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
