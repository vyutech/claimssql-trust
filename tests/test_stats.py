"""P7B tests: bootstrap intervals, the extra report tables, figures and the notebook.

Uses the hand-made run of tests/test_eval.py (15 records, one model "m").
"""

import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from claimssql.eval import bootstrap as bs  # noqa: E402
from claimssql.eval import figures, metrics, report  # noqa: E402
from claimssql.eval.score import score_records  # noqa: E402
from test_eval import RUN, make_record  # noqa: E402

ROOT = Path(__file__).parent.parent
TAUS = {"m": 0.6}


@pytest.fixture(scope="module")
def test_records() -> list[dict]:
    return [make_record(*row) for row in RUN if row[2] == "test"]


# ------------------------------------------------------------- bootstrap ----


def test_stratified_resamples_keep_the_group_sizes():
    groups = ["easy"] * 4 + ["T1"] * 2 + ["hard"] * 3
    index = bs.stratified_indices(groups, 500, np.random.default_rng(1))
    assert index.shape == (500, 9)
    labels = np.array(groups)[index]
    assert (labels[:, :4] == "easy").all() and (labels[:, 4:6] == "T1").all()
    assert (labels[:, 6:] == "hard").all()
    assert len({tuple(row) for row in index}) > 100  # really resampled, with replacement
    assert any(len(set(row[:4])) < 4 for row in index)


def test_point_estimates_equal_the_plain_metrics(test_records):
    result = bs.bootstrap_report(test_records, TAUS, resamples=200, seed=3)
    assert (result["n"], result["resamples"]) == (10, 200)
    for config in ("C0", "C1", "C2", "C3", "C4"):
        plain = metrics.summarise(score_records(test_records, config, 0.6))
        got = result["metrics"][("m", config)]
        for name in ("ex", "coverage", "risk", "rs_0", "rs_1", "rs_10"):
            assert got[name].value == pytest.approx(plain[name]), (config, name)
        harmful = sum(c.region in ("answerable_wrong", "infeasible_answered")
                      for c in score_records(test_records, config, 0.6))  # fmt: skip
        assert got["harmful"].value == harmful
        assert got["harmful"].value == (got["answerable_wrong"].value
                                        + got["infeasible_answered"].value)  # fmt: skip


def test_intervals_are_reproducible_and_contain_the_point(test_records):
    first = bs.bootstrap_report(test_records, TAUS, resamples=300, seed=7)
    again = bs.bootstrap_report(test_records, TAUS, resamples=300, seed=7)
    assert first["metrics"] == again["metrics"] and first["auroc"] == again["auroc"]
    # A different seed draws different resamples.
    groups = ["easy"] * 5 + ["T1"] * 5
    draw = [bs.stratified_indices(groups, 50, np.random.default_rng(seed)) for seed in (7, 7, 8)]
    assert (draw[0] == draw[1]).all() and not (draw[0] == draw[2]).all()
    for estimates in first["metrics"].values():
        for name in ("ex", "coverage", "rs_10", "harmful"):
            est = estimates[name]
            assert est.low <= est.value <= est.high and est.used == 300


def test_a_system_that_is_always_right_has_zero_width_intervals():
    records = [make_record(f"q{i}", "easy", "test", "ok", "ok", 1.0, 90, 10) for i in range(6)]
    result = bs.bootstrap_report(records, TAUS, resamples=100, seed=1)
    c1 = result["metrics"][("m", "C1")]
    assert (c1["ex"].value, c1["ex"].low, c1["ex"].high) == (1.0, 1.0, 1.0)
    assert (c1["harmful"].low, c1["harmful"].high) == (0.0, 0.0)
    assert (c1["rs_10"].low, c1["rs_10"].high) == (100.0, 100.0)
    # AUROC needs right AND wrong answers: undefined here, in every resample.
    auroc = result["auroc"][("m", "1 - agreement")]
    assert (auroc.value, auroc.low, auroc.used) == (None, None, 0)


def test_paired_differences(test_records):
    twin = [{**r, "model_key": "twin"} for r in test_records]
    result = bs.bootstrap_report(
        test_records + twin, {"m": 0.6, "twin": 0.6}, resamples=300, seed=5,
        cross_model=[(("twin", "C4"), ("m", "C4")), (("twin", "C4"), ("m", "C0"))],
    )  # fmt: skip
    found = {(a, b, name): est for a, b, name, est in result["differences"]}
    # Identical systems on the same resamples: the difference is exactly 0, always.
    same = found[("twin C4", "m C4", "rs_10")]
    assert (same.value, same.low, same.high) == (0.0, 0.0, 0.0) and same.excludes_zero() is False
    # C4 - C0 by hand: RS_10 -160 - (-560) = 400; harmful 2 - 6 = -4.
    assert found[("m C4", "m C0", "rs_10")].value == pytest.approx(400.0)
    assert found[("m C4", "m C0", "harmful")].value == -4
    assert found[("twin C4", "m C0", "rs_10")].value == pytest.approx(400.0)
    assert found[("m C4", "m C0", "harmful")].high <= 0


def test_undefined_resamples_are_left_out_and_counted():
    records = [make_record("a", "easy", "test", "bad", "bad", 1.0, 90, 10),
               make_record("b", "easy", "test", "ref", "ref", 0.0, 0, 0)]  # fmt: skip
    result = bs.bootstrap_report(records, TAUS, resamples=400, seed=2)
    risk = result["metrics"][("m", "C1")]["risk"]
    # Both draws hit the refusal a quarter of the time: nothing answered, no risk.
    assert 200 < risk.used < 400 and (risk.value, risk.low, risk.high) == (1.0, 1.0, 1.0)
    assert result["metrics"][("m", "C1")]["coverage"].used == 400


def test_models_must_cover_the_same_questions(test_records):
    partial = [{**r, "model_key": "other"} for r in test_records[:3]]
    with pytest.raises(ValueError, match="same questions"):
        bs.bootstrap_report(test_records + partial, {"m": 0.6, "other": 0.6}, resamples=10)


def test_fast_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        wrong = rng.random(40) < 0.5
        signal = rng.integers(0, 6, 40).astype(float)  # many ties
        if 0 < wrong.sum() < 40:
            assert bs.fast_auroc(wrong, signal) == pytest.approx(roc_auc_score(wrong, signal))
    assert np.isnan(bs.fast_auroc(np.array([True, True]), np.array([1.0, 2.0])))
    assert bs.estimate(float("nan"), np.array([np.nan])).value is None


# ---------------------------------------------------------------- report ----


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("stats")
    runs, out = tmp / "runs", tmp / "tables"
    runs.mkdir()
    records = [make_record(*row) for row in RUN]
    (runs / "fixture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records),
                                        encoding="utf-8")  # fmt: skip
    code = report.main(["--run-id", "fixture", "--runs-dir", str(runs), "--out-dir", str(out),
                        "--low-cardinality", str(tmp / "none.json"), "--resamples", "200",
                        "--seed", "11"])  # fmt: skip
    assert code == 0
    (out / "benchmark_composition.csv").write_text(
        "group,target,count,difference,dev,test,no_split\n"
        "easy,40,4,-36,2,2,0\nmedium,40,3,-37,1,2,0\nhard,40,2,-38,0,2,0\n"
        "T1,25,3,-22,1,2,0\nT2,15,1,-14,0,1,0\nT3,25,1,-24,1,0,0\nT4,15,1,-14,0,1,0\n"
        "total,200,15,-185,5,10,0\n", encoding="utf-8")  # fmt: skip
    return out


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_ci_tables_agree_with_the_point_tables(tables):
    ci = {r["config"]: r for r in read(tables / "e2_rs_ci.csv")}
    plain = {r["config"]: r for r in read(tables / "e2_rs_by_config_penalty.csv")
             if r["split"] == "test"}  # fmt: skip
    for config in ("C0", "C4"):
        for name in ("ex", "coverage", "risk", "rs_0", "rs_1", "rs_10"):
            assert float(ci[config][name]) == pytest.approx(float(plain[config][name]), abs=1e-4)
            assert float(ci[config][f"{name}_lo"]) <= float(ci[config][name])
            assert float(ci[config][name]) <= float(ci[config][f"{name}_hi"])
    assert (ci["C4"]["tau"], ci["C4"]["n"], ci["C0"]["tau"]) == ("0.6", "10", "")

    harmful = {r["config"]: r for r in read(tables / "e3_harmful_ci.csv")}
    assert [harmful[c]["harmful"] for c in ("C0", "C1", "C2", "C3", "C4")] == [
        "6.0", "5.0", "4.0", "3.0", "2.0",
    ]  # fmt: skip
    assert (harmful["C0"]["answerable_wrong"], harmful["C0"]["infeasible_answered"]) == (
        "3.0", "3.0",
    )  # fmt: skip


def test_differences_auroc_and_info_tables(tables):
    differences = {(r["comparison"], r["metric"]): r for r in read(tables / "differences_ci.csv")}
    rs = differences[("m C4 minus m C0", "rs_10")]
    assert (rs["difference"], rs["resamples_used"]) == ("400.0", "200")
    assert differences[("m C4 minus m C0", "harmful")]["difference"] == "-4.0"
    assert len(differences) == 2  # one model: no cross-model comparison

    auroc = {r["signal"]: r for r in read(tables / "e4_auroc_ci.csv")}
    agreement = auroc["1 - agreement"]
    assert (agreement["auroc"], agreement["n_answered"], agreement["n_wrong"]) == ("0.7", "7", "5")
    assert float(agreement["auroc_lo"]) <= 0.7 <= float(agreement["auroc_hi"])

    (info,) = read(tables / "bootstrap_info.csv")
    assert (info["questions"], info["resamples"], info["seed"], info["level"]) == (
        "10", "200", "11", "0.95",
    )  # fmt: skip


def test_figure_data_and_export_tables(tables):
    roc = read(tables / "e4_roc_points.csv")
    assert {r["signal"] for r in roc} == set(bs.SIGNALS)
    first, last = roc[0], [r for r in roc if r["signal"] == roc[0]["signal"]][-1]
    assert (first["false_positive_rate"], first["true_positive_rate"]) == ("0.0", "0.0")
    assert (last["false_positive_rate"], last["true_positive_rate"]) == ("1.0", "1.0")

    groups = {(r["split"], r["group"]): r for r in read(tables / "e7_infeasible_by_group.csv")}
    t1 = groups[("test", "T1")]  # t7 and t8: both answered by C0; C4 answers neither
    assert (t1["n"], t1["answered_C0"], t1["answered_C1"], t1["answered_C4"]) == (
        "2", "2", "1", "0",
    )  # fmt: skip
    assert groups[("test", "T2")]["answered_C4"] == "1"  # t9, agreement 1.0
    assert groups[("all", "T1")]["n"] == "3"

    (cost,) = read(tables / "table7_1_latency_cost.csv")
    # Every question: 1 forced call of 0.2 s and 6 calls of 0.1 s.
    assert (cost["calls_per_question"], cost["p50_seconds_per_call"],
            cost["p95_seconds_per_call"], cost["seconds_per_question_c1_to_c3"],
            cost["seconds_per_question_c4"]) == ("7.0", "0.1", "0.2", "0.1", "0.6")  # fmt: skip
    assert float(cost["mean_seconds_per_call"]) == pytest.approx(0.8 / 7, abs=0.01)
    (model,) = read(tables / "table6_1_models.csv")
    assert (model["model_key"], model["ollama_tag"], model["maker"]) == ("m", "fake-1", "")


def test_real_models_have_report_info():
    from claimssql.config import load_models
    from claimssql.eval.report_stats import cross_model_pairs

    configs = load_models().models
    for key in ("qwen", "gemma", "mid_oss"):
        info = next(c for c in configs if c.key == key).info
        assert {"maker", "parameters", "quantization", "size_gb", "hardware"} <= set(info)
    # RQ4 pairs: each smaller model's C4 against the largest model's C0.
    assert cross_model_pairs(["qwen", "gemma", "mid_oss"], configs) == [
        (("qwen", "C4"), ("mid_oss", "C0")), (("gemma", "C4"), ("mid_oss", "C0")),
    ]  # fmt: skip
    assert cross_model_pairs(["qwen"], configs) == []


# --------------------------------------------------------------- figures ----


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_every_figure_renders_at_300_dpi(name, tables, tmp_path):
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = figures.FIGURES[name](tables)
    path = figures.save(fig, tmp_path / f"{name}.png")
    plt.close(fig)
    assert path.stat().st_size > 5_000
    with Image.open(path) as image:
        assert round(image.info["dpi"][0]) == 300
        assert image.width > 900
    assert all(not ax.get_title() for ax in fig.axes)  # no titles inside the images
    assert all(ax.get_ylabel() or ax.get_xlabel() for ax in fig.axes)


def test_figure_names_and_model_order(tables):
    assert list(figures.FIGURES) == [
        "fig4_2_benchmark_composition", "fig7_1_ex_by_model_difficulty",
        "fig7_2_harmful_by_config", "fig7_3_rs_vs_penalty", "fig7_4_taxonomy_stacked",
        "fig7_5_roc_signals", "fig7_6_risk_coverage", "fig7_7_infeasible_answered",
    ]  # fmt: skip
    assert figures.model_order(tables) == ["m"]
    assert figures.model_labels(tables) == {"m": "fake-1"}


# -------------------------------------------------------------- notebook ----


def test_analysis_notebook_is_valid_and_draws_every_figure():
    notebook = json.loads((ROOT / "notebooks" / "analysis.ipynb").read_text("utf-8"))
    assert notebook["nbformat"] == 4
    code, notes = [], []
    for cell in notebook["cells"]:
        text = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        (code if cell["cell_type"] == "code" else notes).append(text)
    for text in code:
        ast.parse(text)
    everything = "\n".join(code)
    for name in figures.FIGURES:
        assert everything.count(f'"{name}"') == 1, name  # one cell per figure
    assert "full.jsonl" not in everything and "results/runs" not in everything
    assert len(notes) >= len(figures.FIGURES)  # a note for each figure
