"""One-off: MCC of the combined ruleset (min_improvement=0.01) per model, per
dataset. `run()` is the original single-seed (0) beam-search entry point.
`run_seeds()` repeats each dataset over 5 seeds/splits with the greedy
combiner, enforcing <=4 conditions/rule, skipping the 2 largest datasets.
Not part of the CLI.
"""
from __future__ import annotations

import warnings
import signal
import math
import json
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import polars as pl
import numpy as np
from gators.encoders import WOEEncoder
from gators.imputers import NumericImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFClassifier

from imodels import SkopeRulesClassifier

from benchmarks.datasets import load_dataset
from benchmarks.config import FULL_CONFIG
from benchmarks.iguanas_adapter import IguanasAdapter
from benchmarks.baselines import make_baseline, Unavailable
from iguanas.rule_evaluation import apply_rules, apply_and_filter_by_performance
from iguanas.rule_combination import combine_rules_beam_search, combine_rules_greedy
from iguanas.rule_generation import rule_grid_search
from iguanas.weight_transformations import generate_increasing_weights
from iguanas.metrics import count_conditions
from iguanas.rule_selection import filter_rules_by_feature_overlap
from benchmarks.rule_extraction import rules_from_xgboost, rules_from_skope

# Rule pools are expensive to regenerate (26 XGBoost fits per dataset for the
# weight-grid variant); cache them on disk keyed by (dataset, model) so a rerun
# that only changes the combination step (e.g. beam search vs greedy) doesn't
# refit anything. Every model in MODELS below has an identity frame_fn, so only
# the plain rule list needs to be cached.
POOL_CACHE_DIR = Path("/tmp/iguanas_pool_cache")
POOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Same precision/recall floor applied to every model's pool, so no model gets an
# advantage (or handicap) from its own internal filter defaults.
QUALITY_THRESHOLDS = [
    {"name": "precision", "operator": ">=", "value": FULL_CONFIG.selection.min_precision},
    {"name": "recall", "operator": ">=", "value": FULL_CONFIG.selection.min_recall},
]
# Enforced on every candidate rule, for every model, before quality filtering --
# matches the paper's uniform "<=4 conditions per rule" constraint (see
# iguanas_ruleset_benchmark.md section 5.2).
MAX_CONDITIONS = 4
# Diversify the quality-filtered pool down to this many rules before combining,
# to reduce the combiner's "researcher degrees of freedom" on a single split.
DIVERSIFY_TARGET_MIN = 50
DIVERSIFY_TARGET_MAX = 80

# Smallest to largest, by exact row count measured earlier. `pc2` dropped: only
# 23 positives total (16 train / 7 test on a 70/30 split), structurally too
# small for a stable single-split MCC estimate -- see repo memory.
DATASETS = [
    "hepatitis", "ionosphere", "thoracic_surgery", "blood_transfusion", "diabetes",
    "credit_g", "pc1", "pc4", "pc3", "kc1", "ozone_level_8hr", "sick", "spambase",
    "wilt", "churn", "satellite_anomaly", "phoneme", "speeddating", "mc1",
    "jm1", "mammography", "credit_default", "bank_marketing", "adult",
    "aps_failure", "creditcard",
]
# The two largest datasets are excluded from the multi-seed run (run_seeds)
# only: skope_rules_weight_grid's 26 refits already hit the fit timeout on
# these at a single seed, so 5 seeds is not yet affordable for them.
SEED_RUN_SKIP_DATASETS = {"aps_failure", "creditcard"}
MODELS = [
    "iguanas_xgb_all_positive",
    "iguanas_xgb_all_positive_weight_grid",
    "skope_rules",
    "skope_rules_weight_grid",
]
FIT_TIMEOUT_SECONDS = 120
WEIGHT_POWERS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])

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


def _top_feature_weight_grid(X_train, y_train, estimator_params, scale) -> pl.DataFrame:
    """5 power transforms per top-5 feature, plus one Baseline (all-ones) column.

    Including Baseline makes this grid's fitted models a strict superset of the
    single unweighted fit, so the unweighted rule pool is guaranteed to be a
    subset of this pool -- isolating the greedy-selection effect from the
    "different models entirely" effect.
    """
    ranker = XGBClassifier(**estimator_params, scale_pos_weight=scale)
    ranker.fit(X_train.to_pandas(), y_train.astype(int))
    ranked_features = [
        feature
        for feature, _ in sorted(
            zip(X_train.columns, ranker.feature_importances_, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )[:5]
    ]
    frames = [pl.DataFrame({"Baseline": [1.0] * X_train.height})]
    for feature in ranked_features:
        weights = generate_increasing_weights(
            X_train[feature].cast(pl.Float64).abs(), powers=WEIGHT_POWERS
        )
        frames.append(weights.select([column for column in weights.columns if column != "Baseline" and not column.startswith("log(")]))
    return pl.concat(frames, how="horizontal")


def _generate_skope_pool_weight_grid(X_train, y_train, seed: int = 0) -> list[str]:
    """Same top-5-feature x 5-power weight grid as the Iguanas variant, but
    refitting SkopeRulesClassifier per weight column via its native
    sample_weight hook (unlike RuleFit, SkopeRules' fit() accepts it -- see
    _balanced_sample_weight's docstring in baselines.py), unioning rules across
    all 26 fits (1 Baseline + 25 weighted) instead of a single balanced fit.
    """
    pos = float(np.count_nonzero(y_train))
    neg = float(len(y_train) - pos)
    scale = (neg / pos) if pos else 1.0
    ranker_kwargs = dict(
        n_estimators=100, max_depth=4, random_state=seed, tree_method="hist",
        verbosity=0, n_jobs=1, learning_rate=0.3,
    )
    weight_grid = _top_feature_weight_grid(X_train, y_train, ranker_kwargs, scale)
    X_np = X_train.to_pandas().to_numpy()
    y_int = y_train.astype(int)
    columns = list(X_train.columns)
    column_set = set(columns)
    pool: set[str] = set()
    for weight_col in weight_grid.columns:
        sample_weight = weight_grid[weight_col].to_numpy()
        model = SkopeRulesClassifier(
            n_estimators=100, max_depth=4, precision_min=0.0, recall_min=0.0, random_state=seed
        )
        model.fit(X_np, y_int, feature_names=columns, sample_weight=sample_weight)
        pool.update(rules_from_skope(model, column_set))
    return sorted(pool)


def _generate_iguanas_pool(
    estimator_class, leaf_selection: str, X_train, y_train, *, weight_grid: bool = False, seed: int = 0
) -> list[str]:
    """Fit a single balanced-scale_pos_weight model and extract its rule pool.

    Mirrors IguanasAdapter's matched-capacity defaults (n_estimators=100,
    max_depth=4, single balanced scale_pos_weight, no weight-transformation
    grid) so only estimator_class and leaf_selection vary between variants.
    """
    pos = float(np.count_nonzero(y_train))
    neg = float(len(y_train) - pos)
    scale = (neg / pos) if pos else 1.0
    kwargs = dict(
        n_estimators=100,
        max_depth=4,
        random_state=seed,
        tree_method="hist",
        verbosity=0,
        n_jobs=1,
    )
    if estimator_class is XGBClassifier:
        # Matches RuleGenerationConfig.learning_rate; XGBRFClassifier keeps its
        # own bagging-appropriate default instead.
        kwargs["learning_rate"] = 0.3
    estimator = estimator_class(**kwargs)
    sample_weights_df = (
        _top_feature_weight_grid(X_train, y_train, kwargs, scale) if weight_grid else None
    )
    rules_df = rule_grid_search(
        estimator,
        X_train,
        pl.Series(y_train.astype(int)),
        scale_pos_weights=np.array([scale]),
        sample_weights_df=sample_weights_df,
        n_jobs=5 if weight_grid else 1,
        verbose=0,
        leaf_selection=leaf_selection,
    )
    return sorted(set(rules_df["rule"].to_list())) if not rules_df.is_empty() else []


def _pool_and_frame(name: str, X_train, y_train, seed: int = 0):
    if name == "iguanas_xgb_all_positive_weight_grid":
        pool = _generate_iguanas_pool(
            XGBClassifier, "all_positive", X_train, y_train, weight_grid=True, seed=seed
        )
        return pool, lambda X: X
    if name == "skope_rules_weight_grid":
        pool = _generate_skope_pool_weight_grid(X_train, y_train, seed=seed)
        return pool, lambda X: X
    if name in IGUANAS_VARIANTS:
        estimator_class, leaf_selection = IGUANAS_VARIANTS[name]
        pool = _generate_iguanas_pool(estimator_class, leaf_selection, X_train, y_train, seed=seed)
        return pool, lambda X: X
    if name == "iguanas":
        model = IguanasAdapter(FULL_CONFIG, seed=seed)
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
            random_state=seed,
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
    baseline: Any = make_baseline(name, FULL_CONFIG, seed, **kwargs)
    if isinstance(baseline, Unavailable):
        return None, baseline.reason
    baseline.fit_generate(X_train, y_train)
    pool = list(dict.fromkeys(getattr(baseline, "_rules", [])))
    frame_fn = baseline._rule_frame if hasattr(baseline, "_rule_frame") else lambda X: X
    return pool, frame_fn


def _pool_and_frame_cached(dataset_name: str, model_name: str, X_train, y_train, seed: int = 0):
    """Disk-cached wrapper around _pool_and_frame; frame_fn is identity for every
    model currently in MODELS, so only the plain rule list is cached to disk.
    Cache key includes seed so the multi-seed run dumps one pool file per
    (dataset, model, seed) rather than overwriting a single-seed cache entry.
    """
    cache_path = POOL_CACHE_DIR / f"{dataset_name}__{model_name}__seed{seed}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if isinstance(cached, dict) and cached.get("unavailable"):
            return None, cached["unavailable"]
        return cached, (lambda X: X)

    if model_name == "combined_weighted":
        # Union of both weighted pools, reusing their own caches -- no refitting.
        pool_a, reason_a = _pool_and_frame_cached(
            dataset_name, "skope_rules_weight_grid", X_train, y_train, seed=seed
        )
        pool_b, reason_b = _pool_and_frame_cached(
            dataset_name, "iguanas_xgb_all_positive_weight_grid", X_train, y_train, seed=seed
        )
        if pool_a is None and pool_b is None:
            cache_path.write_text(json.dumps({"unavailable": f"{reason_a}; {reason_b}"}))
            return None, f"{reason_a}; {reason_b}"
        pool = sorted(set(pool_a or []) | set(pool_b or []))
        cache_path.write_text(json.dumps(pool))
        return pool, (lambda X: X)

    pool, frame_or_reason = _pool_and_frame(model_name, X_train, y_train, seed=seed)
    if pool is None:
        cache_path.write_text(json.dumps({"unavailable": frame_or_reason}))
        return None, frame_or_reason
    cache_path.write_text(json.dumps(pool))
    return pool, frame_or_reason


def _diversify_pool(metrics: pl.DataFrame) -> list[str]:
    """Shrink a quality-filtered candidate pool to DIVERSIFY_TARGET_MAX rules,
    to reduce the combiner's "researcher degrees of freedom" on a single split.

    Uses filter_rules_by_feature_overlap (keeping the higher-MCC rule from each
    overlapping cluster), escalating min_difference until the survivor count
    drops to DIVERSIFY_TARGET_MAX or below, then keeps the top DIVERSIFY_TARGET_MAX
    by MCC. Pools already at or below the target are returned unchanged.
    """
    if metrics.height <= DIVERSIFY_TARGET_MAX:
        return metrics["rule"].to_list()
    importance = dict(zip(metrics["rule"].to_list(), metrics["mcc"].to_list(), strict=True))
    filtered = metrics
    for min_difference in range(1, 21):
        filtered = filter_rules_by_feature_overlap(metrics, importance, min_difference=min_difference)
        if filtered.height <= DIVERSIFY_TARGET_MAX:
            break
    rules = filtered.sort("mcc", descending=True)["rule"].to_list()
    return rules[:DIVERSIFY_TARGET_MAX]


def _evaluate_dataset(name: str, seed: int, combiner) -> None:
    """Generate -> filter (quality + <=4 conditions) -> combine -> evaluate,
    for one dataset/seed, printing one line per model. combiner is
    combine_rules_greedy or combine_rules_beam_search.
    """
    try:
        dataset = load_dataset(name, max_rows=FULL_CONFIG.max_rows, seed=seed)
    except Exception as exc:
        print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
        return

    idx_train, idx_test = train_test_split(
        range(dataset.X.height), test_size=0.3, random_state=seed, stratify=dataset.y
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

    print(f"--- {name} seed={seed}: {dataset.X.height} rows x {dataset.X.width} features ---")
    for model_name in MODELS:
        try:
            with _Timeout():
                pool, frame_or_reason = _pool_and_frame_cached(
                    name, model_name, X_train, y_train, seed=seed
                )
                if pool is None:
                    print(f"    {model_name:<14} unavailable: {frame_or_reason}")
                    continue
                pool = [rule for rule in pool if count_conditions(rule) <= MAX_CONDITIONS]
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
                diversified = _diversify_pool(metrics)
                R_train = R_train.select(diversified)
                combined = combiner(R_train, y_train_series, metric="mcc", min_improvement=0.01)
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


def run() -> None:
    """Single-seed (0), beam-search entry point. Kept for backward compatibility
    with the earlier greedy-vs-beam comparisons run earlier this session.
    """
    for name in DATASETS:
        _evaluate_dataset(name, seed=0, combiner=combine_rules_beam_search)


def run_seeds(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    """5-seed/split greedy-combiner run, skipping the 2 largest datasets (their
    skope_rules_weight_grid pool generation already hits FIT_TIMEOUT_SECONDS at
    a single seed, so 5 seeds is not yet affordable there).
    """
    for name in DATASETS:
        if name in SEED_RUN_SKIP_DATASETS:
            continue
        for seed in seeds:
            _evaluate_dataset(name, seed=seed, combiner=combine_rules_greedy)


if __name__ == "__main__":
    run()
