"""The rule-set stage every generator shares.

The benchmark compares rule *generators*, so everything downstream of generation
is held constant: the same threshold filter, the same correlation dedup, the same
budgeted maximum-coverage combination, and the same disjunction semantics where a
row is flagged if any selected rule fires. The only thing that varies between
models is which candidate rules they produced.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from iguanas.metrics import compute_metrics, count_conditions
from iguanas.rule_combination import combine_rules_a_star, combine_rules_budgeted
from iguanas.rule_evaluation import apply_and_filter_by_performance
from iguanas.rule_selection import filter_correlated_rules

from .config import ExperimentConfig, SelectionConfig
from .operating_point import select_rule_by_alert_budget


class NoRulesError(ValueError):
    """Raised when no rule survives generation or selection on a fold."""


def select_ruleset(
    frame: pl.DataFrame,
    y: pl.Series,
    rules: list[str],
    cfg: ExperimentConfig,
    sel_cfg: SelectionConfig,
    alert_rate: float,
) -> tuple[str, dict[str, Any]]:
    """Filter, deduplicate and combine candidate rules under an alert budget.

    Returns the chosen rule expression and a report describing how it was
    reached. Raises :class:`NoRulesError` when nothing usable survives.
    """
    if not rules:
        raise NoRulesError("no candidate rules were generated")

    # Boosted ensembles and pruned-tree families repeat identical paths across
    # trees; duplicates would collide as column names during evaluation.
    rules = list(dict.fromkeys(rules))

    # A ruleset should contain no more than 4 conditions per rule (connected by AND).
    if sel_cfg.max_conditions_per_rule is not None:
        rules = [r for r in rules if count_conditions(r) <= sel_cfg.max_conditions_per_rule]

    if not rules:
        raise NoRulesError(f"no candidate rules satisfied max_conditions_per_rule<={sel_cfg.max_conditions_per_rule}")

    thresholds = [
        {"name": "precision", "operator": ">=", "value": sel_cfg.min_precision},
        {"name": "recall", "operator": ">=", "value": sel_cfg.min_recall},
    ]
    R, metrics = apply_and_filter_by_performance(
        frame,
        y,
        rules,
        metric_thresholds=thresholds,
        ranking_metric=sel_cfg.shortlist_metric,
    )
    relaxed = False
    if metrics.is_empty():
        # A generator whose rules all miss the quality floor should be reported as
        # weak, not dropped: excluding the fold removes it from the averages and
        # silently flatters whichever models fail most often. RIPPER and BRL
        # produce very small pools and fail the recall floor outright.
        relaxed = True
        R, metrics = apply_and_filter_by_performance(
            frame,
            y,
            rules,
            metric_thresholds=[],
            ranking_metric=sel_cfg.shortlist_metric,
        )
    if metrics.is_empty():
        raise NoRulesError("no generated rule could be evaluated on the selection split")

    # Shortlist on a coverage-aware metric, then order by precision: at a fixed
    # alert budget the purest rules are the ones that fit inside it.
    ranked = metrics.head(sel_cfg.top_n_rules)
    order_metric = sel_cfg.order_metric
    if order_metric in ranked.columns:
        ranked = ranked.sort(order_metric, descending=True)
    shortlisted = ranked["rule"].to_list()

    if sel_cfg.use_correlation_dedup and len(shortlisted) > 1:
        metric_for_importance = (
            order_metric if order_metric in ranked.columns else cfg.metric
        )
        importance = dict(
            zip(shortlisted, ranked[metric_for_importance].to_list(), strict=True)
        )
        shortlisted = filter_correlated_rules(
            R[shortlisted], importance=importance, max_corr=sel_cfg.max_corr
        )

    # Generators produce wildly different pool sizes -- boosted-path extraction
    # yields hundreds where RIPPER yields three -- so any advantage may simply be
    # more candidates to choose from. Capping equalises what reaches the
    # combination stage and isolates rule quality from pool size.
    if sel_cfg.max_candidate_rules is not None:
        shortlisted = shortlisted[: sel_cfg.max_candidate_rules]

    candidates = R[shortlisted]
    if sel_cfg.exact_budgeted or sel_cfg.budget_metric not in (None, "recall"):
        ruleset = combine_rules_a_star(
            candidates,
            y,
            metric=sel_cfg.budget_metric or cfg.metric,
            max_rules=cfg.max_rules,
            return_top_k=1,
            max_alert_rate=alert_rate,
        )
    else:
        ruleset = combine_rules_budgeted(
            candidates, y, max_alert_rate=alert_rate, max_rules=cfg.max_rules
        )
    if not ruleset.is_empty() and ruleset.columns:
        chosen = ruleset.columns[0]
    else:
        # Nothing fits the budget; fall back to the least-overshooting rule so a
        # usable model is still returned, with its realised rate reported.
        fallback = select_rule_by_alert_budget(
            compute_metrics(candidates, y),
            alert_rate,
            tie_break_metric=sel_cfg.budget_objective,
        )
        if fallback is None:
            raise NoRulesError("no candidate ruleset available within the alert budget")
        chosen = fallback

    report = {
        "n_pool_rules": len(rules),
        "n_candidate_rules": len(shortlisted),
        "n_combined_candidates": int(candidates.width),
        "combiner": (
            f"budgeted_exact:{sel_cfg.budget_metric}"
            if sel_cfg.budget_metric not in (None, "recall")
            else "budgeted_exact" if sel_cfg.exact_budgeted else "budgeted"
        ),
        "thresholds_relaxed": relaxed,
        "chosen_rule": chosen,
    }
    return chosen, report
