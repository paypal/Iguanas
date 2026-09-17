"""Ablation runners.

Each runner varies exactly one design decision and holds everything else — data,
splits, seeds, operating point — fixed, so the resulting rows are directly
comparable within a runner and never across runners.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable

import polars as pl

from .config import ExperimentConfig, RuleGenerationConfig, SelectionConfig
from .datasets import Dataset
from .iguanas_adapter import IguanasAdapter
from .protocol import NestedSplit, make_nested_splits, run_fold


def _run_variant(
    dataset: Dataset,
    cfg: ExperimentConfig,
    splits: Iterable[NestedSplit],
    variant: str,
    arm: str,
    *,
    generation: RuleGenerationConfig,
    selection: SelectionConfig,
    alert_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        adapter = IguanasAdapter(
            cfg,
            seed=split.seed,
            name=f"{variant}:{arm}",
            generation=generation,
            selection=selection,
        )
        result = run_fold(adapter, dataset.X, dataset.y, split, cfg, dataset.name, alert_rate)
        row = result.as_row()
        row.update(
            ablation=variant,
            arm=arm,
            **{f"report_{k}": v for k, v in adapter.selection_report.items()},
        )
        rows.append(row)
    return rows


def _splits_for(dataset: Dataset, cfg: ExperimentConfig) -> list[NestedSplit]:
    return [s for seed in cfg.seeds for s in make_nested_splits(dataset.y, cfg, seed)]


def budget_matched_generation(gen: RuleGenerationConfig) -> RuleGenerationConfig:
    """Weight-steering-off config that fits the same number of trees as steering-on.

    Steering-on fits ``n_scales x n_weight_transformations x n_estimators`` trees.
    Turning steering off collapses the weight axis to a single Baseline column,
    so the scale axis is enlarged by the same factor to keep the budget equal.
    """
    factor = max(1, gen.n_weight_transformations)
    return replace(
        gen,
        use_weight_steering=False,
        n_weight_transformations=0,
        n_scale_pos_weights=gen.n_scale_pos_weights * factor,
    )


def ablation_weight_steering(
    dataset: Dataset, cfg: ExperimentConfig, alert_rate: float
) -> pl.DataFrame:
    splits = _splits_for(dataset, cfg)
    on = replace(cfg.generation, use_weight_steering=True)
    off = budget_matched_generation(on)
    rows = _run_variant(
        dataset, cfg, splits, "weight_steering", "on",
        generation=on, selection=cfg.selection, alert_rate=alert_rate,
    ) + _run_variant(
        dataset, cfg, splits, "weight_steering", "off_budget_matched",
        generation=off, selection=cfg.selection, alert_rate=alert_rate,
    )
    planned = {
        "on": on.n_scale_pos_weights * max(1, on.n_weight_transformations) * on.n_estimators,
        "off_budget_matched": off.n_scale_pos_weights * off.n_estimators,
    }
    return pl.DataFrame(rows).with_columns(
        pl.col("arm").replace_strict(planned, default=None).alias("planned_trees")
    )


def ablation_rule_extraction(
    dataset: Dataset, cfg: ExperimentConfig, alert_rate: float
) -> pl.DataFrame:
    splits = _splits_for(dataset, cfg)
    rows = _run_variant(
        dataset, cfg, splits, "rule_extraction", "max_gain",
        generation=replace(cfg.generation, max_gain_extraction=True),
        selection=cfg.selection, alert_rate=alert_rate,
    ) + _run_variant(
        dataset, cfg, splits, "rule_extraction", "random_path",
        generation=replace(cfg.generation, max_gain_extraction=False),
        selection=cfg.selection, alert_rate=alert_rate,
    )
    return pl.DataFrame(rows)


def ablation_correlation_dedup(
    dataset: Dataset, cfg: ExperimentConfig, alert_rate: float
) -> pl.DataFrame:
    splits = _splits_for(dataset, cfg)
    rows = _run_variant(
        dataset, cfg, splits, "correlation_dedup", "on",
        generation=cfg.generation,
        selection=replace(cfg.selection, use_correlation_dedup=True),
        alert_rate=alert_rate,
    ) + _run_variant(
        dataset, cfg, splits, "correlation_dedup", "off",
        generation=cfg.generation,
        selection=replace(cfg.selection, use_correlation_dedup=False),
        alert_rate=alert_rate,
    )
    return pl.DataFrame(rows)


def ablation_selection_objective(
    dataset: Dataset, cfg: ExperimentConfig, alert_rate: float
) -> pl.DataFrame:
    """Compare coverage-first and precision-first composition on one pool."""
    splits = _splits_for(dataset, cfg)
    rows = _run_variant(
        dataset, cfg, splits, "selection_objective", "coverage_first",
        generation=cfg.generation,
        selection=cfg.selection,
        alert_rate=alert_rate,
    ) + _run_variant(
        dataset, cfg, splits, "selection_objective", "precision_first",
        generation=cfg.generation,
        selection=replace(
            cfg.selection,
            budget_metric="precision",
            exact_budgeted=True,
        ),
        alert_rate=alert_rate,
    )
    return pl.DataFrame(rows)


def ablation_composition_search(
    dataset: Dataset,
    cfg: ExperimentConfig,
    alert_rate: float,
    beam_widths: tuple[int, ...] = (2, 4, 8),
) -> pl.DataFrame:
    """Compare composition strategies, both unconstrained and under the budget.

    The first group maximises a metric with no alert-rate constraint, which is
    what the library's original combiners do. The second group solves the
    budgeted problem the main benchmark actually uses, where
    ``budgeted_greedy`` is the ``1 - 1/e`` approximation and ``budgeted_exact``
    the branch-and-bound optimum; the gap between them is the price of the
    approximation. The two groups are not comparable with each other -- they
    optimise different objectives -- so the arm names keep them distinct.
    """
    splits = _splits_for(dataset, cfg)
    unconstrained = replace(cfg.selection, use_budgeted_combination=False)
    arms: list[tuple[str, SelectionConfig]] = [
        ("greedy", replace(unconstrained, combiner="greedy")),
        *[
            (f"beam_{k}", replace(unconstrained, combiner="beam", beam_width=k))
            for k in beam_widths
        ],
        ("a_star", replace(unconstrained, combiner="a_star")),
        ("exhaustive", replace(unconstrained, combiner="exhaustive")),
        ("budgeted_greedy", replace(cfg.selection, use_budgeted_combination=True)),
        (
            "budgeted_exact",
            replace(cfg.selection, use_budgeted_combination=True, exact_budgeted=True),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for arm, selection in arms:
        rows.extend(
            _run_variant(
                dataset, cfg, splits, "composition_search", arm,
                generation=cfg.generation, selection=selection, alert_rate=alert_rate,
            )
        )
    return pl.DataFrame(rows)


AblationRunner = Callable[[Dataset, ExperimentConfig, float], pl.DataFrame]

ABLATIONS: dict[str, AblationRunner] = {
    "weight_steering": ablation_weight_steering,
    "rule_extraction": ablation_rule_extraction,
    "correlation_dedup": ablation_correlation_dedup,
    "selection_objective": ablation_selection_objective,
    "composition_search": ablation_composition_search,
}


def run_ablations(
    dataset: Dataset,
    cfg: ExperimentConfig,
    alert_rate: float,
    names: Iterable[str] | None = None,
) -> pl.DataFrame:
    selected = list(names) if names is not None else list(ABLATIONS)
    frames = [ABLATIONS[name](dataset, cfg, alert_rate) for name in selected]
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")
