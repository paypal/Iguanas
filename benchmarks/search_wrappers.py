"""Thin, uniform wrappers around the ``iguanas.rule_combination`` search functions.

Rationale: ``combine_rules_a_star`` is under active development and may change
signature or start returning an extra diagnostics object.  Every call site in
the harness goes through :func:`run_combiner`, so adapting to that change is a
one-line edit in :func:`_call_a_star` rather than a sweep through the codebase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from iguanas.rule_combination import (
    combine_rules_a_star,
    combine_rules_beam_search,
    combine_rules_full_search,
    combine_rules_greedy,
)
from iguanas.metrics import compute_metrics

from .timing import Counters

COMBINERS: tuple[str, ...] = ("greedy", "beam", "a_star", "exhaustive")

#: Sentinel for combiners that neither report their own expansions nor admit a
#: closed-form count. Filter these out before averaging node-expansion columns.
NODES_NOT_REPORTED = -1


@dataclass
class CombinationResult:
    """Candidate rulesets produced by a combination search, plus its work counters."""

    R: pl.DataFrame
    metrics: pl.DataFrame
    counters: Counters
    diagnostics: dict[str, Any]


def _unpack(result: Any) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Accept either ``DataFrame`` or ``(DataFrame, diagnostics)`` return shapes."""
    if isinstance(result, pl.DataFrame):
        return result, {}
    if isinstance(result, tuple) and result:
        frame = result[0]
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"combiner returned {type(frame)!r} as first element")
        extra = result[1] if len(result) > 1 else {}
        return frame, dict(extra) if isinstance(extra, dict) else {"diagnostics": extra}
    raise TypeError(f"unsupported combiner return type: {type(result)!r}")


def _call_a_star(
    R: pl.DataFrame,
    y: pl.Series,
    *,
    metric: str,
    max_rules: int,
    operator: str,
    weights: pl.Series | None,
    min_improvement: float | None,
    return_top_k: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    return _unpack(
        combine_rules_a_star(
            R,
            y,
            metric=metric,
            max_rules=max_rules,
            operator=operator,
            weights=weights,
            min_improvement=min_improvement,
            return_top_k=return_top_k,
            return_diagnostics=True,
        )
    )


def _estimate_expansions(combiner: str, n_rules: int, max_rules: int, beam_width: int) -> int:
    """Analytic node-expansion count for combiners that do not report their own.

    These are the exact number of candidate extensions the published algorithms
    enumerate, so they stay comparable across implementations.
    """
    if n_rules == 0:
        return 0
    if combiner == "greedy":
        return sum(max(0, n_rules - depth) for depth in range(min(max_rules, n_rules)))
    if combiner == "beam":
        first = n_rules
        rest = sum(
            min(beam_width, n_rules) * max(0, n_rules - depth)
            for depth in range(1, min(max_rules, n_rules))
        )
        return first + rest
    if combiner == "exhaustive":
        return sum(
            math.comb(n_rules, size) for size in range(2, min(max_rules, n_rules) + 1)
        )
    return NODES_NOT_REPORTED


def run_combiner(
    combiner: str,
    R: pl.DataFrame,
    y: pl.Series,
    *,
    metric: str,
    max_rules: int,
    operator: str = "or",
    weights: pl.Series | None = None,
    min_improvement: float = 0.0,
    return_top_k: int = 10,
    beam_width: int = 4,
) -> CombinationResult:
    """Run one combination search and return candidates with counters attached."""
    if combiner not in COMBINERS:
        raise ValueError(f"combiner must be one of {COMBINERS}, got {combiner!r}")
    if R.is_empty() or not R.columns:
        return CombinationResult(pl.DataFrame(), pl.DataFrame(), Counters(), {})

    diagnostics: dict[str, Any] = {}
    if combiner == "greedy":
        combined = combine_rules_greedy(
            R,
            y,
            metric=metric,
            max_rules=max_rules,
            operator=operator,
            weights=weights,
            min_improvement=min_improvement,
        )
    elif combiner == "beam":
        combined = combine_rules_beam_search(
            R,
            y,
            metric=metric,
            beam_width=beam_width,
            max_rules=max_rules,
            operator=operator,
            weights=weights,
            min_improvement=min_improvement,
            return_top_k=return_top_k,
        )
    elif combiner == "a_star":
        # Always exact: this arm supplies the optimum that the heuristic
        # combiners are measured against, so it must not be pruned heuristically.
        combined, diagnostics = _call_a_star(
            R,
            y,
            metric=metric,
            max_rules=max_rules,
            operator=operator,
            weights=weights,
            min_improvement=None,
            return_top_k=return_top_k,
        )
    else:
        combined = combine_rules_full_search(
            R, n=max_rules, operator=operator
        )

    if combined.is_empty() or not combined.columns:
        return CombinationResult(pl.DataFrame(), pl.DataFrame(), Counters(), diagnostics)

    metrics = compute_metrics(combined, y, weights=weights)
    expansions = int(
        diagnostics.get(
            "nodes_expanded",
            _estimate_expansions(combiner, len(R.columns), max_rules, beam_width),
        )
    )
    counters = Counters(
        nodes_expanded=expansions,
        candidate_sets_evaluated=len(combined.columns),
        metric_evaluations=(
            len(combined.columns)
            if expansions == NODES_NOT_REPORTED
            else expansions + len(combined.columns)
        ),
    )
    return CombinationResult(combined, metrics, counters, diagnostics)
