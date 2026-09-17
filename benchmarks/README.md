# Iguanas benchmark harness

A self-contained experimental harness for evaluating the Iguanas rule-learning
pipeline against interpretable-model baselines under a leakage-free protocol.

Nothing in this directory is imported by the `iguanas` package, and none of its
dependencies are added to `pyproject.toml`.

---

## Quick start

```bash
# from the repository root
python -m pip install -r benchmarks/requirements-bench.txt   # optional baselines
python -m benchmarks.run --smoke                             # ~1-2 minutes
python -m benchmarks.run --full --seeds 5 --ablations        # hours
```

Results land in `benchmarks/results/<mode>-<UTC timestamp>/`.

| CLI flag | Effect |
| --- | --- |
| `--smoke` | 3 small datasets, 1 seed, 3 outer folds, reduced rule pool. Default. |
| `--full` | The entire dataset registry with `FULL_CONFIG`. |
| `--datasets A B` | Explicit registry names, overriding the mode's default list. |
| `--seeds N` | Use seeds `0..N-1`. |
| `--outer-folds N` | Override the outer stratified K-fold count. |
| `--max-rows N` | Stratified, seeded row cap per dataset. |
| `--ablations` | Additionally run all four ablation studies. |
| `--verify-registry` | Probe every dataset entry and print a load report. Runs nothing else. |
| `--out DIR` | Results directory override. |

---

## The evaluation protocol

The harness enforces a **three-way nested split** because the usual two-way
split systematically flatters rule learners: a pool mined from a split has
already fitted that split's noise, and picking the best combination on the same
rows reports a maximum over noise as if it were a generalisation estimate.

```text
outer StratifiedKFold(n_outer_folds)  ->  dev | test
inner stratified split of dev         ->  generate | select
```

* **generate** — the only rows a rule may be mined from.
* **select** — the only rows used for threshold filtering, correlation dedup,
  combination search, operating-point choice and baseline hyper-parameter
  selection.
* **test** — the only rows that appear in a reported number.

`select` receives `1 / n_inner_folds` of the development rows.

### Leakage is checked, not assumed

`benchmarks/protocol.py` builds a `LeakGuard` holding the fold's test indices.
Every frame handed to a model goes through `LeakGuard.subset`, which raises
`LeakageError` if a generation or selection stage requests a test row.
`NestedSplit.validate()` additionally rejects empty, duplicated or overlapping
index sets at construction time. These are runtime assertions, not comments.

### Generalisation gap

For the chosen model, the harness reports

```text
generalisation_gap = metric(select) - metric(test)
```

using one threshold fixed on `select` and applied unchanged to `test`, plus the
same decomposition for precision and recall. `generalisation_gap.csv` is the
headline table.

---

## Matched alert rates

Disjunction learners emit booleans and therefore sit at one uncontrolled point
on the precision/recall curve, while scoring classifiers can be placed anywhere.
Comparing them without matching the **alert rate** (the fraction of the
population flagged for review) compares two different operating regimes.

`benchmarks/operating_point.py` fixes the budget first:

* scoring models — the threshold is the `k`-th largest score on `select`, where
  `k = floor(alert_rate * n)`. Ties are never split, so the *realised* alert rate
  is reported alongside the target.
* rule-based models — the candidate ruleset whose `flagged(%)` is closest to but
  not above the budget is selected, with `metric` as the tie-break. If every
  candidate overshoots, the least-overshooting one is used and its realised rate
  is reported.

Every result row carries `target_alert_rate`, `select_alert_rate` and
`test_alert_rate`. Check them before quoting a precision.

---

## Complexity is counted in conditions

Complexity is always the **number of atomic feature tests**, never the number of
rules. Five rules of six conditions each is not simpler than twelve rules of one
condition each, and rule counts are the usual way that comparison gets fudged.

* trees — the sum over leaves of the path depth (total conditions across all
  decision paths), for both CART and every tree in the GBM.
* rule sets and rule lists — conditions summed over the active rules.

---

## Datasets

`benchmarks/datasets.py` holds a registry of 28 binary classification tasks,
fetched via `sklearn.datasets.fetch_openml` and cached under `benchmarks/.cache/`.
The registry is deliberately weighted towards extreme imbalance (fraud, credit
default, churn, software defect, medical screening), because that is the regime
rule-based alerting is deployed in.

Encoding is minimal and deterministic: non-numeric columns are ordinal-encoded
by sorted category order (so the mapping does not depend on row order), missing
categories become a `__MISSING__` level, and numeric NaNs are median-filled.

### `verified` and the honesty rule

OpenML name/version coordinates drift. Entries whose coordinates have **not**
been confirmed against a live fetch carry `verified=False`. The harness never
guesses a target column or positive label at load time — a mismatch raises
`DatasetLoadError` naming the labels it actually saw.

At the time of writing 27 of the 28 entries verify cleanly, with realised
positive rates within 0.01 of the declared ones. The exception is
`seismic_bumps`, whose `name="seismic-bumps", version=1` coordinates resolve to
a 210-row three-class frame rather than the coal-mine hazard task; it is left in
the registry as `verified=False` with that finding recorded in its `notes`, and
`--full` skips it with a message rather than failing.

To promote an entry, run the probe and edit the registry by hand:

```bash
python -m benchmarks.run --verify-registry
python -m benchmarks.run --verify-registry --datasets creditcard aps_failure
```

The report gives `ok`, the realised positive rate and the error text, so a
mis-declared `positive_label` is visible rather than silently inverted.

`SMOKE` is `("diabetes", "credit_g", "sick")` — small, all verified, and `sick`
supplies a 6% positive rate so the smoke run exercises the imbalanced path.

---

## Baselines

| Name | Source | Notes |
| --- | --- | --- |
| `decision_tree` | scikit-learn | CART pruned along its `ccp_alpha` path; `target_conditions` gives a complexity-matched comparison |
| `gbm_ceiling` | xgboost | Accuracy ceiling, not an interpretable model |
| `rulefit` | imodels | |
| `skope_rules` | imodels | |
| `brl` | imodels | Bayesian Rule Lists; features discretized with `BRLDiscretizer` |
| `figs` | imodels | |
| `corels` | imodels | Needs the optional `corels` wheel; absent from stock imodels |
| `ripper` | wittgenstein | Emits booleans, so scores are 0/1 |

Every optional import is probed, never assumed. A missing package produces a row
in `baseline_availability.csv` with the reason and removes that model from the
run; it never aborts anything. Check that file before concluding a baseline lost.

---

## Ablations

`--ablations` runs four studies, each varying exactly one decision:

1. **`weight_steering`** — on vs off **at equal compute budget**. Steering-on
   fits `n_scales x n_weight_transformations x n_estimators` trees; turning
   steering off collapses the weight axis, so the scale axis is enlarged by the
   same factor (`budget_matched_generation`). Both the planned budget
   (`planned_trees`) and the realised one (`trees_fitted`) are recorded, so the
   match can be audited rather than trusted.
2. **`rule_extraction`** — max-gain path extraction vs a uniformly sampled
   root-to-leaf path from the same trees, over the same scale x weight grid.
3. **`correlation_dedup`** — correlation filtering on vs off.
4. **`composition_search`** — greedy vs beam(2, 4, 8) vs A\* vs exhaustive,
   recording both the achieved metric and the search effort
   (`nodes_expanded`, `candidate_sets_evaluated`).

`nodes_expanded` is `-1` (`search_wrappers.NODES_NOT_REPORTED`) when a combiner
neither reports its own count nor admits a closed form — currently A\*. Filter
that sentinel out before averaging.

---

## Statistics

`benchmarks/reporting.py` emits tidy tables, never plots:

* `pareto_front` — non-dominated (quality, conditions) points per dataset.
* `friedman` — omnibus Friedman test over datasets (requires ≥3 models and ≥3
  datasets; below that it returns a `note` rather than a number).
* `critical_difference` — average ranks plus the Nemenyi critical difference
  from a built-in `q_alpha` table, i.e. everything a CD diagram needs. No extra
  dependency.
* `nemenyi_posthoc` — pairwise p-values via `scikit-posthocs` when installed;
  otherwise a single row saying so.
* `generalisation_gap` — select-minus-test gaps.

Everything is written as CSV, plus parquet when `pyarrow` is importable.

---

## Seed policy

* `ExperimentConfig.seeds` is the only source of randomness. Modules take a seed
  argument; none of them touch a global RNG.
* Seed `s` sets the outer `StratifiedKFold(shuffle=True, random_state=s)`.
* The inner generate/select split of fold `f` uses `random_state = s * 1000 + f`,
  so inner splits differ across folds but are reproducible from `(s, f)` alone.
* Every model is constructed fresh per (seed, fold, alert rate) and receives the
  same seed; no fitted state is reused between folds.
* Dataset sub-sampling under `--max-rows` is stratified and seeded with
  `seeds[0]`.
* Report results over at least 5 seeds for a paper; `--smoke` uses 1 and its
  numbers are for plumbing checks only.

---

## Reproducing a run

Every run directory contains `manifest.json` with the git SHA, full `argv`, the
resolved config (all seeds, folds, alert rates and search parameters), installed
versions of every relevant package, and the hardware capture below. Given that
file the run is reproducible with a single command.

### Hardware disclosure template

`timing.py` records `platform`, `machine`, `processor`, `logical_cpus`,
`total_ram_gb`, and the Python version/implementation into the manifest. Copy the
following into the paper and fill it from `manifest.json`:

```text
All experiments were run on <machine> (<processor>), <logical_cpus> logical
cores, <total_ram_gb> GB RAM, running <platform>, with
<python_implementation> <python_version>. Package versions: iguanas <v>,
numpy <v>, polars <v>, scikit-learn <v>, xgboost <v>, imodels <v>,
wittgenstein <v>. Timings are single-process wall-clock
(RuleGenerationConfig.n_jobs = <n>); alongside them we report the
implementation-independent counters trees_fitted, rules_generated,
nodes_expanded and candidate_sets_evaluated, which are hardware- and
language-agnostic and should be preferred for cross-implementation claims.
```

Wall-clock numbers from a laptop under thermal throttling are not comparable
across papers. The counters are. Quote both.

---

## Module map

| File | Responsibility |
| --- | --- |
| `config.py` | Frozen dataclasses; `SMOKE_CONFIG` and `FULL_CONFIG` |
| `datasets.py` | Registry, OpenML fetch + cache, encoding, `verify_registry` |
| `protocol.py` | Nested splits, `LeakGuard`, per-fold runner, gap computation |
| `operating_point.py` | Matched-alert-rate thresholds and rule-budget selection |
| `baselines.py` | Uniform baseline adapters and availability probing |
| `iguanas_adapter.py` | The Iguanas pipeline as a protocol-compatible model |
| `search_wrappers.py` | Thin uniform wrappers over `iguanas.rule_combination` |
| `ablations.py` | The four ablation runners |
| `reporting.py` | Aggregation, Friedman/Nemenyi, tidy export |
| `timing.py` | Wall-clock, hardware capture, work counters |
| `run.py` | CLI, manifest, orchestration |

### Note on `combine_rules_a_star`

`iguanas.rule_combination.combine_rules_a_star` is being rewritten. Every call
site goes through `search_wrappers.run_combiner`, and the A\* call itself is
isolated in `_call_a_star`. `_unpack` already accepts either a bare `DataFrame`
or a `(DataFrame, diagnostics)` tuple, and reads `nodes_expanded` from the
diagnostics dict when present. If the signature changes, `_call_a_star` is the
one line to edit.
