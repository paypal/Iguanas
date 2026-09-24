Benchmark
=========

This page documents the benchmarking harness that ships with Iguanas in the
``benchmarks/`` directory: the evaluation protocol, the metrics, the baselines,
and the results obtained so far.

.. note::

   The harness is **not** part of the installable package. It carries its own
   dependencies (``benchmarks/requirements-bench.txt``) and is intended for
   reproducing published comparisons, not for production use.

Motivation
----------

Rule systems are almost always deployed under an **alert budget**: only a fixed
fraction of the population can be reviewed, because analyst capacity is fixed.
Accuracy, ROC-AUC and even average precision are poor proxies for that setting.
The harness therefore evaluates every model at *matched alert rates* and reports
complexity alongside quality, so a model is never credited for recall it bought
with an unaffordable alert volume or an unreadable rule set.

Evaluation protocol
-------------------

Three-way nested splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~

Selecting a rule set is itself a search over a large combinatorial space, so
evaluating on the data used to select invites optimism. Each outer fold is split
three ways:

.. code-block:: text

   outer StratifiedKFold  ->  (dev, test)
   inner split of dev     ->  (generate, select)

* **generate** — rules are induced here, and nowhere else.
* **select** — threshold filtering, correlation dedup and rule-set construction.
* **test** — reported metrics only. Never touched before scoring.

A runtime ``LeakGuard`` asserts that no test index reaches the generate or
select stages; it is an executable check, not a comment.

The harness also reports the **generalisation gap**, the difference between the
chosen rule set's metric on *select* and on *test*. This quantifies
selection-induced optimism directly and is treated as a headline result rather
than a diagnostic.

.. warning::

   :func:`iguanas.rule_cv.validate_rules_cv` does **not** implement this
   protocol: it evaluates rules that were generated on the full dataset, so its
   folds are not truly held out and its reported spread is optimistic. Use the
   harness protocol for unbiased estimates.

Matched alert rates
~~~~~~~~~~~~~~~~~~~

Every model is evaluated at a grid of target alert rates. Because rule learners
emit booleans while boosted trees emit scores, the operating point must be
chosen carefully:

* Only thresholds that sit on an **actual score value** are considered, so the
  realised alert rate is one the model can genuinely produce.
* Operating **under** budget is feasible, not a failure. Interpolating a
  quantile instead collapses binary-output models: a rule set firing on 0.9% of
  rows, asked for a 5% alert rate, has its threshold driven to zero and ends up
  flagging the entire population.
* When even the highest-scoring group exceeds the budget, that group is used and
  the overshoot is reported rather than hidden.

The realised alert rate is always recorded next to the target, so comparisons
can be restricted to folds where the budget was actually met.

Metrics
-------

Quality
~~~~~~~

Reported at each matched alert rate: precision, recall, and the realised alert
rate. Average precision is recorded but **not** used for ranking — for a
single-operating-point model it collapses to that model's precision, while
score-based models receive credit for their full ranking, so the two are not
comparable.

Rule quality
~~~~~~~~~~~~

:mod:`iguanas.metrics` provides coverage-aware metrics that a plain precision or
F-score ranking cannot express. With :math:`c` covered instances of :math:`n`,
:math:`p` positives covered and base rate :math:`\pi`:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Definition
     - Why it matters
   * - ``lift``
     - :math:`\frac{p/c}{\pi}`
     - Scale-free, so comparable across datasets with different base rates.
   * - ``wracc``
     - :math:`\frac{c}{n}\left(\frac{p}{c} - \pi\right)`
     - Cannot be gamed by a rule firing on three rows (coverage term vanishes)
       nor by flagging everything (lift term vanishes).
   * - ``laplace``
     - :math:`\frac{p+1}{c+2}`
     - Smoothed precision; penalises low-coverage rules.
   * - ``m_estimate``
     - :math:`\frac{p + m\pi}{c + m}`
     - Shrinks low-coverage rules toward the base rate; ``m`` tunes the strength.

Complexity
~~~~~~~~~~

Complexity is measured in **conditions**, not rules: a three-rule disjunction
with twelve conditions is not simpler than a five-rule one with eight.
:func:`iguanas.metrics.count_conditions` and
:func:`iguanas.metrics.count_features` expose this, and ``compute_metrics``
adds ``num_conditions`` and ``num_features`` columns.

Rule-set construction under a budget
------------------------------------

The operational task is not "maximise F1" but "catch as many positives as
possible without exceeding the alert budget" — the *budgeted maximum coverage*
problem. :func:`iguanas.rule_combination.combine_rules_budgeted` implements the
standard greedy :math:`1 - 1/e` approximation: repeatedly add the feasible rule
with the greatest marginal gain in positives caught, stopping when nothing
feasible adds a positive.

This differs from :func:`iguanas.rule_combination.combine_rules_greedy` in what
it optimises. Metric-greedy combination ignores coverage, so under OR it keeps
widening the rule set until it flags far more than the budget allows. Coverage
is monotone under OR, so a rule that breaches the budget can never be rescued by
adding more rules — which makes the feasibility test a sound pruning rule.

Selection pipeline
~~~~~~~~~~~~~~~~~~

Rules are filtered **before** they are combined, which matters because OR-composition
only widens coverage:

1. **Threshold filter** on the select split (default ``precision >= 0.5``,
   ``recall >= 0.05``).
2. **Shortlist** the top-N by WRAcc.
3. **Order** by precision, which also drives correlation-dedup importance.
4. **Construct** the rule set with budgeted maximum coverage.

Baselines
---------

All baselines are adapted to a common interface and report complexity in
conditions. Optional dependencies degrade to an "unavailable" record rather than
failing the run.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Baseline
     - Notes
   * - ``rulefit``
     - Rule ensemble with L1 sparsification (``imodels``).
   * - ``skope_rules``
     - Precision/recall-filtered rule ensemble (``imodels``).
   * - ``brl``
     - Bayesian Rule Lists (``imodels``).
   * - ``figs``
     - Fast Interpretable Greedy-tree Sums (``imodels``).
   * - ``corels``
     - Certifiably optimal rule lists; needs the optional ``corels`` wheel.
   * - ``ripper``
     - Sequential covering (``wittgenstein``).
   * - ``decision_tree``
     - CART pruned via ``ccp_alpha`` to matched complexity.
   * - ``gbm_ceiling``
     - Gradient-boosted trees as an *uninterpretable* performance ceiling.

Datasets
--------

The registry favours **extreme class imbalance**, the regime where the alert
budget framing applies. Sub-sampling preserves the base rate and enforces a
floor on retained positives (``MIN_POSITIVES``): a purely proportional cap
annihilates the minority class, leaving for example seven positives on
``creditcard``, at which point every model degenerates to predicting
all-positive. ``max_rows`` is therefore advisory.

Datasets with too few positives to support any threshold — ``pc2`` has 23 in
total — are unusable regardless of sampling and should be excluded.

Results
-------

Mammography, nested protocol, matched alert budgets, three outer folds. Base
rate 2.3%. Complexity is mean conditions in the final model.

**Alert budget 5%**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15

   * - Model
     - Alert rate
     - Recall
     - Precision
     - Conditions
   * - ``rulefit``
     - 0.053
     - 0.790
     - 0.354
     - 62.3
   * - ``gbm_ceiling``
     - 0.040
     - 0.785
     - 0.466
     - 13139
   * - **iguanas**
     - **0.032**
     - **0.665**
     - **0.476**
     - **9.0**
   * - ``figs``
     - 0.037
     - 0.571
     - 0.387
     - 12.0
   * - ``skope_rules``
     - 0.031
     - 0.550
     - 0.473
     - 39.0

**Alert budget 1%**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15

   * - Model
     - Alert rate
     - Recall
     - Precision
     - Conditions
   * - ``ripper``
     - 0.018
     - 0.465
     - 0.592
     - 46.7
   * - ``gbm_ceiling``
     - 0.013
     - 0.456
     - 0.875
     - 13139
   * - ``rulefit``
     - 0.012
     - 0.425
     - 0.864
     - 62.3
   * - ``skope_rules``
     - 0.010
     - 0.375
     - 0.860
     - 39.0
   * - **iguanas**
     - **0.010**
     - **0.360**
     - **0.819**
     - **5.0**

Against ``skope_rules`` at matched alert rate and comparable precision, Iguanas
gains **+0.115 recall at the 5% budget with 4.3x fewer conditions**, and is
statistically tied at the 1% budget with **8x fewer conditions**. Both remain
below the uninterpretable ``gbm_ceiling``, which is the expected interpretability
cost.

.. important::

   These figures are **one dataset over three folds** and are not a claim of
   general superiority. A multi-dataset run with repeated seeds, Friedman tests
   and critical-difference diagrams is required before drawing conclusions; the
   harness produces those tables but they are not reported here yet.

Reproducing
-----------

.. code-block:: bash

   pip install -r benchmarks/requirements-bench.txt

   # fast end-to-end check on three small datasets
   python -m benchmarks.run --smoke

   # the whole registry
   python -m benchmarks.run --full

   # a specific subset, with ablations
   python -m benchmarks.run --smoke --ablations --datasets mammography creditcard

Each run writes a directory under ``benchmarks/results/`` containing tidy CSV and
Parquet tables plus a ``manifest.json`` capturing the git SHA, every seed, the
resolved package versions and the host's hardware. Runs are reproducible from the
manifest alone.

Ablations
---------

``--ablations`` adds four studies:

* **Weight steering** on versus off at *equal compute budget*, with the realised
  number of trees recorded so the match can be verified.
* **Rule extraction**: max-gain path versus a uniformly sampled root-to-leaf path.
* **Correlation dedup** on versus off.
* **Composition search**: greedy, beam(k), exact branch-and-bound and exhaustive,
  recording achieved metric *and* node expansions.

Threats to validity
-------------------

* Score-emitting models can reach any alert rate by moving a threshold, whereas a
  rule learner can only use coverage levels present in its pool. Matched-alert-rate
  comparison is fair in intent but still favours score emitters.
* Iguanas extracts one max-gain rule per tree; RuleFit and SkopeRules extract every
  root-to-leaf path. Matching on trees therefore does *not* match on candidate
  rules, and the harness matches on candidate rules while recording trees separately.
* Raising the rule-set size cap gives the budgeted greedy more freedom to overfit
  the select split; the generalisation gap should be read alongside any gain.
