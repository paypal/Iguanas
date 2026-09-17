"""Regression tests for the alert-budget operating point.

These guard the two defects that silently invalidated whole benchmark runs:
quantile thresholding collapsing binary-output models, and the budget selector
optimising coverage instead of quality.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from benchmarks.operating_point import (
    choose_threshold,
    predictions_at_alert_rate,
    select_rule_by_alert_budget,
)
from benchmarks.timing import FitTimeout, time_limit


class TestTimeLimit:
    """A timeout that a library can swallow is not a timeout."""

    def test_is_not_caught_by_a_broad_exception_handler(self):
        """sklearn's ``_fit_and_score`` catches ``Exception`` and carries on."""
        assert issubclass(FitTimeout, BaseException)
        assert not issubclass(FitTimeout, Exception)

        with pytest.raises(FitTimeout):
            with time_limit(0.05):
                try:
                    while True:
                        pass
                except Exception:  # noqa: BLE001 - emulating a third-party handler
                    pytest.fail("FitTimeout was swallowed by `except Exception`")

    def test_fast_block_is_untouched(self):
        with time_limit(30):
            result = sum(range(100))
        assert result == 4950

    def test_alarm_is_cleared_after_the_block(self):
        """A leaked itimer would fire during unrelated later work."""
        with time_limit(0.05):
            pass
        import time as _time

        _time.sleep(0.2)  # would raise here if the alarm were still armed

    @pytest.mark.parametrize("disabled", [None, 0, -1])
    def test_disabled_budget_is_a_no_op(self, disabled):
        with time_limit(disabled):
            assert True


class TestChooseThreshold:
    def test_binary_scores_under_budget_are_not_inflated(self):
        """A ruleset firing on 0.9% asked for 5% must stay at 0.9%, not flag everything."""
        scores = np.array([1.0] * 9 + [0.0] * 991)
        point = choose_threshold(scores, 0.05)

        assert point.realised_alert_rate == pytest.approx(0.009)
        assert point.n_flagged == 9

    def test_binary_scores_over_budget_report_the_overshoot(self):
        scores = np.array([1.0] * 9 + [0.0] * 991)
        point = choose_threshold(scores, 0.005)

        assert point.realised_alert_rate == pytest.approx(0.009)
        assert point.realised_alert_rate > point.target_alert_rate

    def test_continuous_scores_hit_the_budget(self):
        scores = np.random.default_rng(0).random(1000)
        assert choose_threshold(scores, 0.05).realised_alert_rate == pytest.approx(0.05)

    def test_threshold_sits_on_an_actual_score_value(self):
        scores = np.random.default_rng(1).random(500)
        point = choose_threshold(scores, 0.1)
        assert point.threshold in set(scores.tolist())

    def test_predictions_match_the_reported_count(self):
        scores = np.array([1.0] * 20 + [0.0] * 80)
        pred, point = predictions_at_alert_rate(scores, 0.5)
        assert int(pred.sum()) == point.n_flagged

    def test_constant_scores_never_flag_everything_silently(self):
        pred, point = predictions_at_alert_rate(np.zeros(100), 0.05)
        assert point.realised_alert_rate == 1.0  # only one achievable point
        assert int(pred.sum()) == 100

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_invalid_alert_rate_raises(self, bad):
        with pytest.raises(ValueError, match="alert_rate"):
            choose_threshold(np.array([1.0, 0.0]), bad)


class TestSelectRuleByAlertBudget:
    @staticmethod
    def _metrics(rows):
        return pl.DataFrame(rows)

    def test_prefers_quality_over_filling_the_budget(self):
        """The broadest rule that fits is usually the worst one that fits."""
        metrics = self._metrics(
            [
                {"rule": "broad_but_weak", "flagged(%)": 0.99, "wracc": 0.001},
                {"rule": "sharp", "flagged(%)": 0.50, "wracc": 0.020},
            ]
        )
        assert select_rule_by_alert_budget(metrics, 0.01, tie_break_metric="wracc") == "sharp"

    def test_ignores_candidates_over_budget(self):
        metrics = self._metrics(
            [
                {"rule": "too_broad", "flagged(%)": 40.0, "wracc": 0.90},
                {"rule": "fits", "flagged(%)": 0.80, "wracc": 0.01},
            ]
        )
        assert select_rule_by_alert_budget(metrics, 0.01, tie_break_metric="wracc") == "fits"

    def test_falls_back_to_least_overshoot(self):
        metrics = self._metrics(
            [
                {"rule": "huge", "flagged(%)": 90.0, "wracc": 0.5},
                {"rule": "smallest", "flagged(%)": 12.0, "wracc": 0.1},
            ]
        )
        assert select_rule_by_alert_budget(metrics, 0.01, tie_break_metric="wracc") == "smallest"

    def test_empty_metrics_returns_none(self):
        assert select_rule_by_alert_budget(pl.DataFrame(), 0.05) is None

    def test_missing_flagged_column_raises(self):
        with pytest.raises(KeyError, match="flagged"):
            select_rule_by_alert_budget(self._metrics([{"rule": "a", "wracc": 1.0}]), 0.05)
