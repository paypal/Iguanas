"""One-off: F1 of the greedy-combined ruleset (min_improvement=0.01) per model,
one 70/30 stratified split per dataset, no alert-rate sweep. Not part of the CLI.
"""
from __future__ import annotations

import warnings
import signal
import math

warnings.filterwarnings("ignore")

import polars as pl
import numpy as np
from gators.encoders import WOEEncoder
from gators.imputers import NumericImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFClassifier

from benchmarks.datasets import load_dataset
from benchmarks.config import FULL_CONFIG
from benchmarks.iguanas_adapter import IguanasAdapter
from benchmarks.baselines import make_baseline, Unavailable
from iguanas.rule_evaluation import apply_rules, apply_and_filter_by_performance
from iguanas.rule_combination import combine_rules_greedy
from iguanas.rule_generation import rule_grid_search
from benchmarks.rule_extraction import rules_from_xgboost

# Same precision/recall floor applied to every model's pool, so no model gets an
# advantage (or handicap) from its own internal filter defaults.
QUALITY_THRESHOLDS = [
    {"name": "precision", "operator": ">=", "value": FULL_CONFIG.selection.min_precision},
    {"name": "recall", "operator": ">=", "value": FULL_CONFIG.selection.min_recall},
]

# Smallest to largest, by exact row count measured earlier.
DATASETS = [
    "hepatitis", "ionosphere", "thoracic_surgery", "blood_transfusion", "diabetes",
    "credit_g", "pc1", "pc4", "pc3", "kc1", "ozone_level_8hr", "sick", "spambase",
    "wilt", "churn", "satellite_anomaly", "phoneme", "pc2", "speeddating", "mc1",
    "jm1", "mammography", "credit_default", "bank_marketing", "adult",
    "aps_failure", "creditcard",
]
MODELS = [
    "iguanas_xgb_max_gain", "iguanas_xgb_all_positive",
    "iguanas_xgbrf_max_gain", "iguanas_xgbrf_all_positive",
    "skope_rules",
]
FIT_TIMEOUT_SECONDS = 120

# (estimator class, leaf_selection) per Iguanas-style generation variant.
IGUANAS_VARIANTS = {
    "iguanas_xgb_max_gain": (XGBClassifier, "max_gain"),
    "iguanas_xgb_all_positive": (XGBClassifier, "all_positive"),
    "iguanas_xgbrf_max_gain": (XGBRFClassifier, "max_gain"),
    "iguanas_xgbrf_all_positive": (XGBRFClassifier, "all_positive"),
}


class _FitTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _FitTimeout


class _Timeout:
    def __enter__(self):
        self._previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, FIT_TIMEOUT_SECONDS)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._previous_handler)
        return False


def _generate_iguanas_pool(estimator_class, leaf_selection: str, X_train, y_train) -> list[str]:
    """Fit a single balanced-scale_pos_weight model and extract its rule pool.

    Mirrors IguanasAdapter's matched-capacity defaults (n_estimators=100,
    max_depth=4, single balanced scale_pos_weight, no weight-transformation
    grid) so only estimator_class and leaf_selection vary between variants.
    """
    pos = float(np.count_nonzero(y_train))
    neg = float(len(y_train) - pos)
    scale = (neg / pos) if pos else 1.0
    kwargs = dict(n_estimators=100, max_depth=4, random_state=0, tree_method="hist", verbosity=0)
    if estimator_class is XGBClassifier:
        # Matches RuleGenerationConfig.learning_rate; XGBRFClassifier keeps its
        # own bagging-appropriate default instead.
        kwargs["learning_rate"] = 0.3
    estimator = estimator_class(**kwargs)
    rules_df = rule_grid_search(
        estimator,
        X_train,
        pl.Series(y_train.astype(int)),
        scale_pos_weights=np.array([scale]),
        n_jobs=1,
        verbose=0,
        leaf_selection=leaf_selection,
    )
    return sorted(set(rules_df["rule"].to_list())) if not rules_df.is_empty() else []


def _pool_and_frame(name: str, X_train, y_train):
    if name in IGUANAS_VARIANTS:
        estimator_class, leaf_selection = IGUANAS_VARIANTS[name]
        pool = _generate_iguanas_pool(estimator_class, leaf_selection, X_train, y_train)
        return pool, lambda X: X
    if name == "iguanas":
        model = IguanasAdapter(FULL_CONFIG, seed=0)
        model.fit_generate(X_train, y_train)
        return list(dict.fromkeys(model._pool.rules)), lambda X: X
    if name == "xgbrf_ceiling":
        # Bagged (not boosted) XGBoost trees: each tree fits the raw labels
        # independently, like SkopeRules' random-forest generator.
        pos = float(np.count_nonzero(y_train))
        neg = float(len(y_train) - pos)
        model = XGBRFClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=0,
            tree_method="hist",
            scale_pos_weight=(neg / pos) if pos else 1.0,
            verbosity=0,
        )
        model.fit(X_train.to_pandas(), y_train.astype(int))
        pool = list(dict.fromkeys(rules_from_xgboost(model.get_booster().trees_to_dataframe())))
        return pool, lambda X: X
    kwargs = {}
    if name == "skope_rules":
        # imodels' own defaults (precision_min=0.5, recall_min=0.01) don't match
        # QUALITY_THRESHOLDS; disable them here so the pool is filtered once,
        # identically to every other model, instead of twice with mismatched cuts.
        kwargs = {"precision_min": 0.0, "recall_min": 0.0}
    model = make_baseline(name, FULL_CONFIG, 0, **kwargs)
    if isinstance(model, Unavailable):
        return None, model.reason
    model.fit_generate(X_train, y_train)
    pool = list(dict.fromkeys(getattr(model, "_rules", [])))
    frame_fn = model._rule_frame if hasattr(model, "_rule_frame") else lambda X: X
    return pool, frame_fn


def run() -> None:
    for name in DATASETS:
        try:
            dataset = load_dataset(name, max_rows=FULL_CONFIG.max_rows, seed=0)
        except Exception as exc:
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            continue

        idx_train, idx_test = train_test_split(
            range(dataset.X.height), test_size=0.3, random_state=0, stratify=dataset.y
        )
        X_train, y_train = dataset.X[idx_train], dataset.y[idx_train]
        X_test, y_test = dataset.X[idx_test], dataset.y[idx_test]
        imputer = NumericImputer(strategy="mean", inplace=True)
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)
        encoder = WOEEncoder(inplace=True)
        y_train_series = pl.Series("target", y_train.astype(bool))
        encoder.fit(X_train, y_train_series)
        X_train = encoder.transform(X_train).fill_nan(0).fill_null(0)
        X_test = encoder.transform(X_test).fill_nan(0).fill_null(0)
        y_test_series = pl.Series(y_test.astype(bool))

        print(f"--- {name}: {dataset.X.height} rows x {dataset.X.width} features ---")
        for model_name in MODELS:
            try:
                with _Timeout():
                    pool, frame_or_reason = _pool_and_frame(model_name, X_train, y_train)
                    if pool is None:
                        print(f"    {model_name:<14} unavailable: {frame_or_reason}")
                        continue
                    if not pool:
                        print(f"    {model_name:<14} no rules generated")
                        continue
                    frame_fn = frame_or_reason
                    train_frame = frame_fn(X_train)
                    R_train, metrics = apply_and_filter_by_performance(
                        train_frame, y_train_series, pool, metric_thresholds=QUALITY_THRESHOLDS
                    )
                    if metrics.is_empty():
                        # Nothing survives the floor: report as weak rather than dropping the
                        # model, matching select_ruleset's own relaxation fallback.
                        R_train, metrics = apply_and_filter_by_performance(
                            train_frame, y_train_series, pool, metric_thresholds=[]
                        )
                    if metrics.is_empty():
                        print(f"    {model_name:<14} no rule could be evaluated")
                        continue
                    combined = combine_rules_greedy(
                        R_train, y_train_series, metric="mcc", min_improvement=0.01
                    )
                    rule_expr = combined.columns[0]

                    # Evaluate that fixed rule expression on the held-out test split.
                    test_pred_df = apply_rules(frame_fn(X_test), [rule_expr])
                    pred = test_pred_df[rule_expr]
                    tp = int((pred & y_test_series).sum())
                    fp = int((pred & ~y_test_series).sum())
                    fn = int((~pred & y_test_series).sum())
                    tn = int((~pred & ~y_test_series).sum())
                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall = tp / (tp + fn) if (tp + fn) else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
                    n_selected = rule_expr.count(") | (") + 1
                    print(
                        f"    {model_name:<14} pool={len(pool):>5} filtered={R_train.width:>5} "
                        f"selected={n_selected} mcc={mcc:.3f} f1={f1:.3f} precision={precision:.3f} recall={recall:.3f}"
                    )
            except _FitTimeout:
                print(f"    {model_name:<14} TIMEOUT after {FIT_TIMEOUT_SECONDS}s; continuing")
            except Exception as exc:
                print(f"    {model_name:<14} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run()
