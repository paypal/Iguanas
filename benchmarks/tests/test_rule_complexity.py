"""Tests ensuring rulesets contain no more than 4 conditions per rule (connected by AND)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from benchmarks.baselines import make_baseline
from benchmarks.config import SMOKE_CONFIG, SelectionConfig
from benchmarks.datasets import load_dataset
from benchmarks.iguanas_adapter import IguanasAdapter
from benchmarks.preprocessing import SharedPreprocessor
from benchmarks.ruleset_selection import select_ruleset
from iguanas.metrics import count_conditions


class TestMaxConditionsPerRule:
    """Every rule in a ruleset must contain no more than 4 conditions connected by AND."""

    def test_select_ruleset_filters_rules_exceeding_max_conditions(self):
        # Rule with 5 conditions
        rule_5 = '(X["a"] > 1) & (X["b"] > 2) & (X["c"] > 3) & (X["d"] > 4) & (X["e"] > 5)'
        # Rule with 3 conditions
        rule_3 = '(X["a"] > 1) & (X["b"] > 2) & (X["c"] > 3)'
        assert count_conditions(rule_5) == 5
        assert count_conditions(rule_3) == 3

        frame = pl.DataFrame({
            "a": [2.0, 0.0, 2.0, 0.0],
            "b": [3.0, 0.0, 3.0, 0.0],
            "c": [4.0, 0.0, 0.0, 0.0],
            "d": [5.0, 0.0, 5.0, 0.0],
            "e": [6.0, 0.0, 6.0, 0.0],
        })
        y = pl.Series([True, False, True, False])
        sel_cfg = SelectionConfig(max_conditions_per_rule=4, min_precision=0.0, min_recall=0.0)

        chosen, report = select_ruleset(
            frame, y, [rule_5, rule_3], SMOKE_CONFIG, sel_cfg, alert_rate=0.5
        )
        # Only rule_3 should be shortlisted and chosen
        assert rule_5 not in chosen
        assert count_conditions(chosen) <= 4

    def test_all_models_produce_rules_with_at_most_4_conditions(self):
        ds = load_dataset("mammography", max_rows=1000, seed=0)
        pre = SharedPreprocessor(discretize=False).fit(ds.X)
        X_trans = pre.transform(ds.X)
        y = ds.y

        models = ["decision_tree", "gbm_ceiling", "rulefit", "skope_rules", "figs"]
        for name in models:
            b = make_baseline(name, SMOKE_CONFIG, 0)
            b.fit_generate(X_trans, y)
            rules = b._extract_rules()
            for r in rules:
                assert count_conditions(r) <= 4, f"{name} produced rule with >4 conditions: {r}"

        ig = IguanasAdapter(SMOKE_CONFIG, 0)
        ig.fit_generate(X_trans, y)
        for r in ig._pool.rules:
            assert count_conditions(r) <= 4, f"Iguanas produced rule with >4 conditions: {r}"
