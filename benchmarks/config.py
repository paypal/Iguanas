"""Experiment configuration objects.

All randomness in the harness is derived from :attr:`ExperimentConfig.seeds`;
no module is allowed to call a global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
CACHE_DIR = BENCH_ROOT / ".cache"
RESULTS_DIR = BENCH_ROOT / "results"


@dataclass(frozen=True)
class RuleGenerationConfig:
    """Controls the Iguanas rule-pool generation stage.

    Iguanas extracts ONE max-gain rule per tree, whereas RuleFit and SkopeRules
    extract every root-to-leaf path (~2**max_depth rules per tree). Matching the
    baselines on *trees* would therefore starve the candidate pool by an order of
    magnitude, so the grid is sized to match on candidate rules instead; trees
    fitted is recorded separately as the compute measure.

    Defaults are matched-capacity, no-grid settings: a single fit at the
    class_weight='balanced' equivalent (see
    :meth:`IguanasAdapter._scale_pos_weights`), ``n_estimators=100`` and
    ``max_depth=4``, so model capacity is not a confound when comparing against
    ``gbm_ceiling`` (same n_estimators/max_depth/balanced scale_pos_weight) or
    the imodels baselines (same n_estimators/tree_size where supported).
    """

    n_scale_pos_weights: int = 1
    n_weight_transformations: int = 0
    weight_feature_mode: str = "variance"
    n_weight_features: int = 5
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.3
    use_weight_steering: bool = True
    max_gain_extraction: bool = True
    n_jobs: int = 1


@dataclass(frozen=True)
class SelectionConfig:
    """Controls threshold filtering, dedup and combination search.

    Thresholds are applied to individual rules on the *select* split before any
    dedup or combination. This ordering matters: OR-composition only ever widens
    coverage, so admitting low-precision rules to the pool makes the combined
    rule set overshoot any alert budget.
    """

    min_precision: float = 0.3
    min_recall: float = 0.15
    max_conditions_per_rule: int = 4
    top_n_rules: int = 150
    max_candidate_rules: int | None = None
    shortlist_metric: str = "wracc"
    order_metric: str = "precision"
    budget_objective: str = "wracc"
    budget_metric: str | None = None
    max_corr: float = 0.9
    use_correlation_dedup: bool = True
    combiner: str = "greedy"
    use_budgeted_combination: bool = True
    exact_budgeted: bool = False
    beam_width: int = 4
    return_top_k: int = 10
    min_improvement: float = 0.0
    combine_operator: str = "or"
    full_search_depth: int = 3


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    ``n_inner_folds`` defines the *generate*/*select* split of each outer
    development fold: ``select`` receives ``1 / n_inner_folds`` of the dev rows.
    """

    seeds: tuple[int, ...] = (0, 1, 2)
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    metric: str = "f1"
    selection_split: str = "train"
    # Densely sampled so each model's operating curve is resolved rather than
    # represented by a couple of points, most of which saturate.
    alert_rates: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
    primary_alert_rate: float = 0.05
    max_rules: int = 10
    fit_timeout_s: float = 900.0
    max_rows: int = 60_000
    baselines: tuple[str, ...] = (
        "gbm_ceiling",
        "rulefit",
        "skope_rules",
        "brl",
        "figs",
        "corels",
        "ripper",
    )
    generation: RuleGenerationConfig = field(default_factory=RuleGenerationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    results_dir: Path = RESULTS_DIR

    def with_seed_count(self, n: int) -> ExperimentConfig:
        return replace(self, seeds=tuple(range(n)))


SMOKE_CONFIG = ExperimentConfig(
    seeds=(0,),
    n_outer_folds=3,
    n_inner_folds=3,
    metric="f1",
    alert_rates=(0.005, 0.01, 0.02, 0.03, 0.05, 0.10),
    primary_alert_rate=0.05,
    max_rules=8,
    fit_timeout_s=120.0,
    max_rows=4_000,
    generation=RuleGenerationConfig(
        n_scale_pos_weights=1,
        n_weight_transformations=0,
        n_estimators=100,
        max_depth=4,
    ),
    selection=SelectionConfig(top_n_rules=80, full_search_depth=2),
)

FULL_CONFIG = ExperimentConfig()
