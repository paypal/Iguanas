"""Aggregation, statistical testing and tidy result export.

Outputs are tables first and plots never: every artefact written here is a tidy
CSV (plus parquet when pyarrow is present) that a paper's figure script can read
without re-running the experiments.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

# Studentised-range critical values q_alpha / sqrt(2) for the Nemenyi test,
# indexed by number of compared models (Demsar 2006, Table 5).
_NEMENYI_Q05: dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
    12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
}
_NEMENYI_Q10: dict[int, float] = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
    7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978,
    12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159,
}

SUCCESS = "ok"


def successful(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty() or "status" not in results.columns:
        return results
    return results.filter(pl.col("status") == SUCCESS)


def summarise_by_model(
    results: pl.DataFrame, value_column: str, group: tuple[str, ...] = ("dataset", "model")
) -> pl.DataFrame:
    """Mean/std of ``value_column`` across seeds and folds."""
    df = successful(results)
    if df.is_empty() or value_column not in df.columns:
        return pl.DataFrame()
    return (
        df.group_by(list(group))
        .agg(
            pl.col(value_column).mean().alias(f"{value_column}_mean"),
            pl.col(value_column).std().alias(f"{value_column}_std"),
            pl.len().alias("n_runs"),
        )
        .sort(list(group))
    )


def pareto_front(
    results: pl.DataFrame,
    *,
    quality_column: str = "test_average_precision",
    cost_column: str = "complexity_conditions",
) -> pl.DataFrame:
    """Non-dominated (max quality, min cost) points, one row per model/dataset."""
    df = successful(results)
    if df.is_empty() or quality_column not in df.columns:
        return pl.DataFrame()
    agg = (
        df.group_by(["dataset", "model"])
        .agg(
            pl.col(quality_column).mean().alias("quality"),
            pl.col(cost_column).mean().alias("cost"),
        )
        .drop_nulls(["quality", "cost"])
    )
    fronts: list[pl.DataFrame] = []
    for (dataset,), group in agg.group_by(["dataset"]):
        quality = group["quality"].to_numpy()
        cost = group["cost"].to_numpy()
        dominated = np.zeros(len(group), dtype=bool)
        for i in range(len(group)):
            dominated[i] = bool(
                np.any((quality >= quality[i]) & (cost <= cost[i]) & (
                    (quality > quality[i]) | (cost < cost[i])
                ))
            )
        fronts.append(
            group.with_columns(
                pl.Series("on_pareto_front", ~dominated),
                pl.lit(dataset).alias("dataset"),
            )
        )
    return pl.concat(fronts).sort(["dataset", "cost"])


def rank_matrix(
    results: pl.DataFrame, value_column: str, *, higher_is_better: bool = True
) -> tuple[pl.DataFrame, list[str]]:
    """Datasets x models table of mean scores, restricted to fully covered models."""
    summary = summarise_by_model(results, value_column)
    if summary.is_empty():
        return pl.DataFrame(), []
    wide = summary.pivot(
        values=f"{value_column}_mean", index="dataset", on="model"
    ).drop_nulls()
    models = [c for c in wide.columns if c != "dataset"]
    if not models:
        return pl.DataFrame(), []
    sign = -1.0 if higher_is_better else 1.0
    scores = wide.select(models).to_numpy() * sign
    ranks = np.apply_along_axis(stats.rankdata, 1, scores)
    ranked = pl.DataFrame(ranks, schema=models).with_columns(wide["dataset"])
    return ranked, models


def friedman_test(results: pl.DataFrame, value_column: str) -> dict[str, Any]:
    """Friedman omnibus test over datasets, comparing models on ``value_column``."""
    summary = summarise_by_model(results, value_column)
    if summary.is_empty():
        return {"statistic": float("nan"), "p_value": float("nan"), "n_models": 0}
    wide = summary.pivot(
        values=f"{value_column}_mean", index="dataset", on="model"
    ).drop_nulls()
    models = [c for c in wide.columns if c != "dataset"]
    if len(models) < 3 or wide.height < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_models": len(models),
            "n_datasets": wide.height,
            "note": "Friedman needs >=3 models and >=3 datasets",
        }
    columns = [wide[m].to_numpy() for m in models]
    statistic, p_value = stats.friedmanchisquare(*columns)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_models": len(models),
        "n_datasets": int(wide.height),
        "models": ",".join(models),
    }


def nemenyi_critical_difference(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    table = _NEMENYI_Q05 if alpha <= 0.05 else _NEMENYI_Q10
    if n_models not in table or n_datasets < 1:
        return float("nan")
    q = table[n_models]
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets)))


def nemenyi_posthoc(results: pl.DataFrame, value_column: str) -> pl.DataFrame:
    """Pairwise Nemenyi p-values; uses scikit-posthocs when installed."""
    ranked, models = rank_matrix(results, value_column)
    if not models:
        return pl.DataFrame()
    try:
        sp = importlib.import_module("scikit_posthocs")
    except ImportError:
        return pl.DataFrame(
            {"note": ["scikit-posthocs unavailable; use critical_difference_data instead"]}
        )
    matrix = sp.posthoc_nemenyi_friedman(ranked.select(models).to_pandas())
    long = matrix.stack().reset_index()
    long.columns = ["model_a", "model_b", "p_value"]
    return pl.from_pandas(long)


def critical_difference_data(
    results: pl.DataFrame, value_column: str, alpha: float = 0.05
) -> pl.DataFrame:
    """Average ranks plus the critical difference — everything a CD diagram needs."""
    ranked, models = rank_matrix(results, value_column)
    if not models:
        return pl.DataFrame()
    n_datasets = ranked.height
    cd = nemenyi_critical_difference(len(models), n_datasets, alpha)
    avg_ranks = [float(np.mean(ranked[m].to_numpy())) for m in models]
    return pl.DataFrame(
        {
            "model": models,
            "average_rank": avg_ranks,
            "n_datasets": [n_datasets] * len(models),
            "n_models": [len(models)] * len(models),
            "critical_difference": [cd] * len(models),
            "alpha": [alpha] * len(models),
            "metric": [value_column] * len(models),
        }
    ).sort("average_rank")


def operating_curve(results: pl.DataFrame) -> pl.DataFrame:
    """Each model's realised alert rate against what it caught there.

    Rule sets cannot be dialled to an arbitrary alert rate the way a scorer can:
    they occupy only the coverage levels their rules produce, and they *saturate*
    once no further candidate fits. Comparing two models at the same nominal
    budget therefore often compares them at different points on their own
    curves, which flatters whichever one stopped earlier at a high-precision
    point. This table is the raw material for comparing at matched coverage.
    """
    df = successful(results)
    if df.is_empty():
        return pl.DataFrame()
    return (
        df.group_by(["dataset", "model", "target_alert_rate"])
        .agg(
            pl.col("test_alert_rate").mean().alias("realised_alert_rate"),
            pl.col("test_recall").mean().alias("recall"),
            pl.col("test_precision").mean().alias("precision"),
            pl.col("complexity_conditions").mean().alias("conditions"),
            pl.col("budget_utilisation").mean().alias("budget_utilisation"),
            pl.len().alias("n_runs"),
        )
        .with_columns(
            pl.when(pl.col("realised_alert_rate") > 0)
            .then(pl.col("recall") / pl.col("realised_alert_rate"))
            .otherwise(0.0)
            .alias("recall_per_alert")
        )
        .sort(["dataset", "model", "realised_alert_rate"])
    )


def saturation_table(results: pl.DataFrame) -> pl.DataFrame:
    """Where each model stops being able to spend more of the alert budget."""
    curve = operating_curve(results)
    if curve.is_empty():
        return pl.DataFrame()
    return (
        curve.group_by(["dataset", "model"])
        .agg(
            pl.col("realised_alert_rate").max().alias("max_alert_rate"),
            pl.col("realised_alert_rate").n_unique().alias("distinct_operating_points"),
            pl.col("target_alert_rate").n_unique().alias("targets_requested"),
            pl.col("budget_utilisation").max().alias("best_budget_utilisation"),
        )
        .with_columns(
            (pl.col("distinct_operating_points") < pl.col("targets_requested")).alias("saturates")
        )
        .sort(["dataset", "max_alert_rate"], descending=[False, True])
    )


def matched_alert_rate_comparison(
    results: pl.DataFrame, tolerance: float = 0.25
) -> pl.DataFrame:
    """Compare models only where their realised alert rates actually coincide.

    For every dataset and target budget, models whose realised rate lies within
    *tolerance* of the group median are kept; the rest are reported but flagged,
    because a model sitting at half the coverage of its peers is on a different
    part of its curve and its precision/recall are not comparable.
    """
    curve = operating_curve(results)
    if curve.is_empty():
        return pl.DataFrame()
    return (
        curve.with_columns(
            pl.col("realised_alert_rate")
            .median()
            .over(["dataset", "target_alert_rate"])
            .alias("group_median_alert_rate")
        )
        .with_columns(
            (
                (pl.col("realised_alert_rate") - pl.col("group_median_alert_rate")).abs()
                <= tolerance * pl.col("group_median_alert_rate")
            ).alias("comparable")
        )
        .sort(["dataset", "target_alert_rate", "recall"], descending=[False, False, True])
    )


def generalisation_gap_table(results: pl.DataFrame) -> pl.DataFrame:
    """Selection-minus-test gaps: the headline evidence for the nested protocol."""
    df = successful(results)
    if df.is_empty() or "generalisation_gap" not in df.columns:
        return pl.DataFrame()
    return (
        df.group_by(["dataset", "model"])
        .agg(
            pl.col("generalisation_gap").mean().alias("gap_mean"),
            pl.col("generalisation_gap").std().alias("gap_std"),
            pl.col("precision_gap").mean().alias("precision_gap_mean"),
            pl.col("recall_gap").mean().alias("recall_gap_mean"),
            pl.col("test_alert_rate").mean().alias("test_alert_rate_mean"),
            pl.len().alias("n_runs"),
        )
        .sort(["dataset", "gap_mean"], descending=[False, True])
    )


def write_table(frame: pl.DataFrame, directory: Path, stem: str) -> list[Path]:
    """Write a frame as CSV and, when pyarrow is available, parquet."""
    if frame.is_empty():
        return []
    directory.mkdir(parents=True, exist_ok=True)
    written = [directory / f"{stem}.csv"]
    frame.write_csv(written[0])
    try:
        importlib.import_module("pyarrow")
    except ImportError:
        return written
    parquet_path = directory / f"{stem}.parquet"
    frame.write_parquet(parquet_path)
    written.append(parquet_path)
    return written


def build_report(
    results: pl.DataFrame, directory: Path, *, metric_column: str = "test_average_precision"
) -> dict[str, list[Path]]:
    """Write every derived table for a completed run."""
    tables: dict[str, pl.DataFrame] = {
        "raw_results": results,
        "summary_quality": summarise_by_model(results, metric_column),
        "summary_complexity": summarise_by_model(results, "complexity_conditions"),
        "pareto_front": pareto_front(results, quality_column=metric_column),
        "critical_difference": critical_difference_data(results, metric_column),
        "nemenyi_posthoc": nemenyi_posthoc(results, metric_column),
        "generalisation_gap": generalisation_gap_table(results),
        "operating_curve": operating_curve(results),
        "saturation": saturation_table(results),
        "matched_alert_rate": matched_alert_rate_comparison(results),
        "friedman": pl.DataFrame([friedman_test(results, metric_column)]),
    }
    return {name: write_table(frame, directory, name) for name, frame in tables.items()}
