# Iguanas: Diversity-Controlled Generation and Alert-Budgeted Composition of Disjunctive Rule Sets

## Abstract

Rule-based classifiers are attractive when predictions must be inspected,
approved, and deployed as explicit policy. In many such applications, however,
the number of alerts that can be reviewed is limited and rules must remain concise
and readable. A useful rule system must therefore control not only predictive quality,
but also alert volume, rule-level complexity (e.g. at most 4 conditions per rule
connected by AND), and the compute spent searching for candidate rules. We present
Iguanas as a framework for this problem. Iguanas generates a diverse candidate pool
by crossing `scale_pos_weight` values with feature-derived sample-weight schedules,
filters and deduplicates the resulting rules (enforcing $\le 4$ atomic conditions
per rule), and composes them as a disjunction under an explicit alert budget. We
formulate budgeted composition as maximum positive coverage subject to an
alert-volume constraint and provide both a greedy approximation and an exact
branch-and-bound solver with an admissible upper bound for common classification
metrics.

The paper separates two decisions that are often conflated: **generation-grid
design**, which determines the available precision-recall-compute frontier, and
**composition objective**, which chooses a deployable point on that frontier.
Under the constraint of $\le 4$ conditions per rule across 11 datasets and three
seeds, replacing the variance-based sample-weight grid with a top-five-important-feature
grid changes mean precision from $0.611 \pm 0.202$ to $0.620 \pm 0.203$ at a
matched six-scale budget, while mean F1 changes from $0.480 \pm 0.210$ to
$0.486 \pm 0.218$. Expanding the scale grid from six to ten values gives mean
recall $0.461 \pm 0.258$ and mean F1 $0.488 \pm 0.213$, at approximately 73%
more fitted trees. Separately, precision-first composition raises mean precision
from 0.613 to 0.766, while mean recall falls from 0.465 to 0.272; these two arms
do not realise the same alert rate and are therefore an objective-sensitivity
analysis, not a matched operating-point comparison. In a 15-dataset benchmark
where all models are constrained to rules with at most 4 conditions, FIGS and
SkopeRules are strong precision-oriented comparators, while Iguanas provides a
wider recall and alert-coverage range. The evidence supports Iguanas as a
controllable framework for operational rule-set design, not as a universally
dominant classifier.

## Highlights

- Every rule in a ruleset is constrained to at most 4 atomic conditions connected by AND.
- Alert budgets define deployable operating points for rule systems.
- Class and sample weighting control candidate-rule diversity.
- Exact search composes rules under an explicit alert constraint.
- Feature-targeted weighting improves precision at matched compute.
- Nested evaluation separates generation, selection, and testing.

## Keywords

Interpretable machine learning; rule-based classification; alert budget;
imbalanced classification; rule ensembles; branch-and-bound; explainable AI

## 1. Introduction

Rule systems are useful when a prediction must be explained as a concrete set of conditions. An analyst can inspect a rule, a policy owner can approve it, and a deployment service can evaluate it without retaining a probabilistic model. This property is valuable in fraud, abuse, compliance, risk, and other workflows in which predictions trigger human action. The distinction between inherently interpretable models and post-hoc explanations is important in high-stakes settings [1].

These workflows impose a constraint that is easy to miss in conventional classification evaluation: review capacity is finite. The relevant question is not simply whether a model ranks positive examples highly. It is how many positives the resulting rule set can recover at each affordable alert rate, how many conditions the analyst must maintain, and how much computation was required to obtain the candidate rules.

The distinction is especially important for disjunctive rule systems. If a row is flagged whenever any selected rule fires, adding a rule can recover new positives but also add false positives. A small rule set may be highly precise but saturate at low coverage. A larger rule set may spend more of the alert budget and recover more positives while reducing precision. Neither behavior is universally preferable; the appropriate operating point depends on the cost of review and the relative cost of missed positives.

Iguanas addresses this problem through a pipeline with two controllable search layers:

1. **Diversity-controlled generation:** fit boosted-tree models under a grid of class-weight values and sample-weight schedules, then extract interpretable decision paths into a candidate pool.
2. **Alert-budgeted composition:** filter weak and redundant rules, then select a disjunctive ruleset subject to a maximum alert rate using either greedy or exact search.

The central contribution is not a new tree-induction algorithm. Tree-path extraction from boosted trees is established [2]. The contribution is the explicit separation of generation and composition as measurable design problems. The generation grid determines which trade-offs are available; the composition objective determines which feasible trade-off is deployed.

The contributions are:

- an alert-budgeted evaluation protocol for boolean rule emitters;
- a generation framework that treats class weighting and feature-derived sample weighting as complementary diversity controls;
- a maximum-coverage formulation for OR-composed rulesets under an alert budget;
- an exact branch-and-bound solver with an admissible bound for precision, recall, accuracy, and F-beta metrics;
- a nested benchmark that separates induction, selection, composition, and test evaluation;
- an empirical analysis showing how generation-grid design and composition objective affect precision, recall, complexity, and compute;
- a calibrated comparison against FIGS, SkopeRules, RuleFit, boosted paths, decision trees, RIPPER, and Bayesian Rule Lists.

The work is positioned between rule ensembles and certifiably optimized rule
models. RuleFit provides a canonical rule-ensemble construction [3], Bayesian
Rule Lists provide an interpretable ordered-rule formulation [4], and CORELS
and GOSDT study certified optimization for rule lists and sparse trees [5,6].
The broader rule-learning literature includes early separate-and-conquer
learners such as RIPPER [8], decision-set models that jointly optimize
interpretability and prediction [9], and falling rule lists for ordered risk
stratification [10]. Tree-ensemble rule extraction has also been studied through
inTrees [11], stable rule-set construction through SIRUS [12], and optimization
approaches such as FIRE [13]. These methods establish important alternatives in
which rules are induced, extracted, or optimized under different structural
assumptions. Iguanas addresses a different setting: a generated pool of
potentially useful rules is composed as an alert-budgeted disjunction, with
explicit controls for candidate diversity and compute. The distinction between
interpretable model classes and post-hoc rule extraction is also central to
responsible interpretable machine learning [14]. The closest formulation is the
Boolean decision-rule literature, which directly studies disjunctive normal form
rules and constrained rule selection [16], and Bayesian Rule Sets, which model
conjunctions and their disjunctive combination probabilistically [17]. WRAcc
also comes from subgroup discovery, where coverage-aware quality measures and
diverse subgroup collections are central objects [18]. Iguanas should therefore
be viewed as a software-and-protocol contribution that instantiates and
evaluates these ideas under a specific alert-budgeted generation pipeline, not
as the first method to learn a DNF rule set.

## 2. Problem Formulation

Let $X$ be a population of $n$ instances and let $y \in \{0,1\}^n$ be the target.
Each candidate rule $r_j$ is a conjunction of atomic conditions (feature tests):

$$
r_j(x) = \bigwedge_{l=1}^{L_j} c_{jl}(x),
$$

where the number of conditions per rule is constrained by an upper bound $L_{\max} = 4$
(i.e., $L_j \le 4$). A selected ruleset $S$ uses disjunctive normal form (DNF) semantics:

$$
\hat y_i(S) = \bigvee_{j \in S} r_{ij}.
$$

The realised alert rate is

$$
\operatorname{AR}(S) = \frac{1}{n}\sum_{i=1}^{n}\hat y_i(S).
$$

For a maximum alert rate $\alpha$ and a maximum of $k$ selected rules, the coverage-first composition problem is

$$
\max_{S: |S| \leq k,\ \operatorname{AR}(S) \leq \alpha,\ \forall j \in S: L_j \le 4}
\sum_{i=1}^{n} y_i \hat y_i(S).
$$

The objective counts covered positives. It is appropriate when review capacity is fixed and the operational goal is to recover as many positives as possible within that capacity. Precision-first and F-beta variants replace the objective while retaining the same alert constraint.

For OR composition, adding a rule can only increase TP and FP. Let $P$ denote the total positive mass and let $g_j$ be the additional positive mass supplied by a remaining rule. If $s$ rule slots remain, an optimistic upper bound is

$$
TP_{\max} = \min\left(P, TP + \sum_{j=1}^{s} g_{(j)}\right),
$$

where $g_{(j)}$ are the largest remaining gains. The bound assumes that these gains are disjoint and that no additional false positives are introduced. It is therefore optimistic. Since recall, precision, accuracy, and F-beta are non-decreasing in TP and non-increasing in FP, evaluating the metric at this optimistic state gives an admissible branch-and-bound bound. The alert budget further limits the additional covered mass to the unused capacity.

## 3. Diversity-Controlled Generation

### 3.1 Class-weight and sample-weight steering

Iguanas fits a grid of boosted-tree models under different `scale_pos_weight` values and sample-weight transformations. Each fitted tree contributes a high-gain root-to-leaf path as an interpretable rule. The underlying booster is XGBoost [2].

The two weighting mechanisms change different parts of the fitting process:

- `scale_pos_weight` changes the relative cost of minority-class errors;
- sample weights change which observations contribute most strongly to the fitted splits.

Their Cartesian product is used as a **rule-generation grid**, not as a conventional hyperparameter search in which one model is selected by validation score. Rules from the fitted models are pooled and deduplicated. The purpose is to induce useful diversity in the candidate pool.

The scale values are logarithmically spaced on the generation split:

```python
ratio = max(2.0, negatives / positives)
scale_pos_weights = np.logspace(0.0, np.log10(ratio), num=n_scales)
```

Thus the grid starts at 1.0 and ends at the negative-to-positive ratio, with logarithmic spacing. A six-scale, four-weight grid is abbreviated as 6x4 and fits 24 model configurations. With 15 trees per configuration, this is 360 trees before accounting for failed or unavailable fits. A 10x4 grid fits 40 configurations and approximately 600 trees.

### 3.2 Sample-weight schedules

For a numerical feature $x$, `generate_weights` creates increasing and decreasing schedules based on powers and logarithms. The default power set is

$$
\{0.25, 0.5, 1, 2, 4\}.
$$

The increasing family contains $(1+x)^p$ and $\log(1+x)$; the decreasing family contains $1/(1+x)^p$ and an inverse logarithmic schedule. The feature is shifted by its minimum before transformation.

The benchmark compares two feature-selection modes:

- **variance mode:** use the highest-variance feature as the basis for the sample-weight grid;
- **top-five mode:** fit a class-balanced importance model on the generation split, select its five most important features, generate schedules from all five, and retain the requested number of decorrelated schedules.

The feature-importance model is fitted only on the generation split. No test labels are used to select features or weighting schedules.

### 3.3 Selection and deduplication

Candidate rules are evaluated on a separate selection split. To enforce readability
and domain interpretability, each individual rule is strictly constrained to contain
at most 4 atomic conditions connected by AND ($L_j \le 4$). Any candidate exceeding
this bound is discarded before threshold filtering.

Rules must normally meet minimum precision and recall thresholds. They are shortlisted
using WRAcc, which combines enrichment and coverage:

$$
WRAcc = \frac{c}{n}\left(\frac{p}{c} - \pi\right),
$$

where $c$ is covered population, $p$ is covered positive mass, and $\pi$ is the population positive rate. Highly correlated candidates are removed before composition. This prevents a large number of near-duplicate paths from being mistaken for useful generation diversity.

## 4. Alert-Budgeted Composition

### 4.1 Coverage-first composition

The default composer repeatedly adds the feasible candidate with the greatest marginal gain in positive coverage. A candidate is infeasible when its union with the current ruleset exceeds the alert budget. Under OR composition, coverage is monotone, so an over-budget ruleset cannot become feasible by adding more rules and its descendant branch can be pruned.

This greedy method is a standard maximum-coverage approximation. It is useful as a fast operational method, but it does not optimize precision or F1 directly.

### 4.2 Exact branch-and-bound composition

The exact solver explores subsets of the fixed, generated candidate pool in
best-first order. Each partial subset has a metric upper bound. A branch is
pruned when its bound cannot beat the current incumbent or when its alert volume
makes all descendants infeasible. The solver is exact **conditional on that
fixed candidate pool and objective** when no heuristic minimum-improvement filter
is used and the objective is one of the supported bounded metrics. It does not
claim global optimality over all possible rule expressions, which would require
jointly optimizing rule induction and composition as in a different model class.

On mammography, exact budgeted composition achieved selection F1 of approximately 0.621 compared with 0.547 for budgeted greedy composition. This is an improvement of approximately 0.074 on the selection split; it is evidence about composition-search loss, not an unbiased test-set superiority claim. The alert constraint also pruned the search, making the exact solver a reference for modest candidate pools and a useful way to quantify heuristic composition loss.

## 5. Experimental Protocol

### 5.1 Nested evaluation

Each outer fold is divided into three disjoint portions:

- **generate:** fit the rule generator and its weighting grid;
- **select:** filter, rank, deduplicate, and compose rules;
- **test:** evaluate the frozen ruleset exactly once.

Preprocessing is fitted without test data. A runtime leak guard verifies that test indices do not enter generation or selection. Generation is cached across alert-rate targets because the candidate pool does not depend on the requested budget.

### 5.2 Baselines and semantics

We compare Iguanas with decision trees, boosted-tree paths, RuleFit [3],
SkopeRules, FIGS [7], RIPPER, and Bayesian Rule Lists [4]. To make the main
comparison explicit, all systems are evaluated as disjunctions: a row is
flagged if any selected rule fires. Furthermore, the critical condition of
**at most 4 conditions per rule** (connected by AND) is uniformly enforced
across all models during rule extraction and selection.

This common semantics is useful for comparing rule generators, but it is not native to every baseline. In particular, Bayesian Rule Lists are ordered models; flattening an ordered list into an OR ruleset changes its semantics. Such models are reported separately and failure rates are retained rather than hidden.

### 5.3 Metrics

We report precision, recall, F0.5, F1, F2, MCC, WRAcc, realised alert rate, condition count, generated-rule count, and fitted-tree count. F0.5 emphasises precision; F2 emphasises recall. Average precision is recorded but not used to rank boolean emitters because it is nearly equivalent to precision for a single operating point, while score-based systems receive credit for their full ranking.

Every model is evaluated over a grid of target alert rates. Since boolean rulesets cannot necessarily realise every nominal rate, each result records the realised alert rate. Matched-alert comparisons use realised rather than requested coverage.

### 5.4 Datasets and configurations

The broad benchmark covered 15 public tabular datasets using one seed, three outer folds, six alert-rate targets, and a stratified row cap where appropriate. The high-fidelity mammography benchmark used three seeds, five outer folds, the full alert-rate grid, and the full generation configuration.

The generation-grid ablation covered 11 datasets, three seeds, three outer folds, and a 5% target. It compared:

- current variance-based feature selection with a 6x4 grid;
- top-five importance selection with a matched 6x4 grid;
- top-five importance selection with an expanded 10x4 grid.

## 6. Results

### 6.1 Dataset-by-dataset operating points

The primary baseline comparison is shown at the nominal 5% alert budget. Each cell reports **realised alert rate / precision / recall / F1**. This view is preferable to a single pooled leaderboard because it reveals saturation, dataset heterogeneity, and the different operating behavior of precision-first and coverage-oriented systems.

| Dataset | Iguanas | FIGS | SkopeRules | Boosted paths |
|---|---|---|---|---|
| bank_marketing | 4.9% / .480 / .203 / .283 | 3.9% / .586 / .190 / .280 | 4.7% / .595 / .235 / .336 | 5.3% / .474 / .214 / .293 |
| churn | 5.0% / .902 / .316 / .468 | 4.5% / .873 / .279 / .420 | 4.5% / .762 / .240 / .365 | 4.7% / .772 / .260 / .389 |
| credit_default | 4.7% / .686 / .145 / .237 | 5.1% / .599 / .136 / .220 | 5.5% / .718 / .179 / .285 | 5.0% / .641 / .145 / .236 |
| creditcard | 0.5% / .327 / .865 / .472 | 0.2% / .713 / .731 / .697 | 0.2% / .801 / .800 / .798 | 0.3% / .443 / .840 / .579 |
| jm1 | 4.7% / .499 / .120 / .193 | 3.2% / .523 / .085 / .143 | 5.0% / .465 / .120 / .191 | 5.0% / .444 / .115 / .183 |
| kc1 | 5.7% / .574 / .212 / .309 | 5.0% / .607 / .196 / .296 | 4.9% / .571 / .181 / .274 | 5.2% / .591 / .199 / .297 |
| mammography | 5.3% / .325 / .735 / .449 | 2.1% / .626 / .550 / .584 | 2.3% / .546 / .525 / .529 | 4.7% / .380 / .760 / .506 |
| mc1 | 0.9% / .381 / .440 / .402 | 0.3% / .776 / .264 / .369 | 0.3% / .690 / .249 / .356 | 0.7% / .447 / .397 / .403 |
| ozone_level_8hr | 4.8% / .383 / .281 / .322 | 3.6% / .354 / .207 / .256 | 3.9% / .368 / .232 / .282 | 4.5% / .375 / .262 / .308 |
| pc1 | 3.8% / .397 / .208 / .269 | 3.4% / .493 / .222 / .293 | 2.8% / .501 / .209 / .293 | 4.5% / .351 / .220 / .265 |
| pc3 | 5.7% / .358 / .200 / .256 | 3.1% / .307 / .081 / .126 | 2.8% / .376 / .100 / .157 | 6.0% / .378 / .218 / .265 |
| pc4 | 5.1% / .713 / .304 / .422 | 3.9% / .769 / .247 / .374 | 4.7% / .659 / .259 / .369 | 5.7% / .673 / .321 / .434 |
| satellite_anomaly | 2.4% / .464 / .733 / .565 | 1.3% / .783 / .587 / .634 | 1.2% / .727 / .573 / .615 | 2.3% / .488 / .747 / .590 |
| sick | 4.9% / .930 / .736 / .821 | 5.9% / .747 / .732 / .738 | 6.3% / .828 / .853 / .840 | 4.8% / .825 / .649 / .726 |
| wilt | 4.9% / .737 / .671 / .700 | 4.4% / .782 / .639 / .700 | 4.9% / .775 / .699 / .729 | 5.1% / .676 / .634 / .654 |

FIGS is usually purer and often saturates below 5% coverage. The table is not a matched-alert comparison: in several rows Iguanas spends substantially more alert volume than FIGS. It therefore describes each method's realised operating behavior rather than proving recall superiority. Matched-alert comparisons and operating curves are the appropriate basis for claims about relative quality. SkopeRules is competitive when its compact pool reaches the requested operating point. Boosted paths provide a high-recall reference but usually require more conditions.

### 6.2 Precision-first versus coverage-first composition

The controlled objective ablation uses the same Iguanas candidate pool for both composition arms. Only the composition objective changes.

| Composition objective | Mean realised alert rate | Precision | Recall | F1 | F0.5 | F2 |
|---|---:|---:|---:|---:|---:|---:|
| Coverage-first | 3.58% | 0.613 | 0.465 | 0.485 | 0.533 | 0.467 |
| Precision-first | 1.48% | 0.766 | 0.272 | 0.353 | 0.467 | 0.298 |

Precision-first improved precision on 8 of 11 datasets. Coverage-first improved recall on all 11. Because the arms realise different mean alert rates (1.48% versus 3.58%), this is an objective-sensitivity result rather than a fair matched-coverage ranking. It does demonstrate that the lower precision of the default Iguanas results is not an intrinsic inability to generate precise rules: it is associated with selecting a ruleset that spends more of the available alert budget to recover additional positives.

### 6.3 Generation-grid ablation

The complete generation-grid experiment included 99 fold-level results per arm. Values below are means over datasets, seeds, and folds with standard deviations over fold-level test results. The six-scale arms fit approximately the same number of trees; the ten-scale arm deliberately expands the class-weight axis.

| Generation grid | Mean trees | Mean generated rules | Precision | Recall | F1 | F2 |
|---|---:|---:|---:|---:|---:|---:|
| Variance feature, 6x4 | 346.4 | 174.6 | 0.611 ± 0.202 | 0.456 ± 0.254 | 0.480 ± 0.210 | 0.460 ± 0.234 |
| Top five important features, 6x4 | 359.1 | 196.7 | 0.620 ± 0.203 | 0.460 ± 0.261 | 0.486 ± 0.218 | 0.464 ± 0.242 |
| Top five important features, 10x4 | 598.5 | 288.0 | 0.616 ± 0.191 | 0.461 ± 0.258 | 0.488 ± 0.213 | 0.466 ± 0.239 |

At seed 0, the top-five 6x4 arm achieved the highest precision on 6 of 11 datasets, but these exploratory win counts are not treated as inferential evidence. Across all three seeds, the top-five arms show only small mean differences relative to fold-level variability. The 10x4 arm costs about 73% more fitted trees than the current arm; its mean F1 advantage over the current arm is approximately 0.008, with overlapping variability.

These results support treating feature selection and scale-grid size as separate experimental factors, but do not establish a statistically significant winner. Top-five targeting is a plausible precision-oriented alternative at a matched six-scale budget. The expanded ten-scale grid is a higher-cost configuration with a small average F1/F2 advantage in this experiment. The current variance grid remains a more economical default until a larger multi-seed study or a prespecified statistical test supports changing it. No single generation grid dominates across all datasets and objectives.

### 6.4 FIGS as a precision-first comparator

At the 5% target in the 15-dataset smoke benchmark with $\le 4$ conditions per rule,
FIGS achieved mean precision $0.636$ and mean F0.5 $0.498$, while SkopeRules achieved
mean precision $0.625$ and mean F0.5 $0.511$. These models produce compact, high-precision
rulesets.

Iguanas had mean recall $0.399$ (and F2 $0.399$) with mean F1 $0.411$, while FIGS achieved
mean recall $0.363$ and mean F2 $0.363$. These differences should not be read as
matched-alert superiority across all operating points. The correct comparison is therefore
not “Iguanas versus FIGS as a single winner.” FIGS is a compact precision-first system;
Iguanas exposes a broader coverage-oriented design space and can recover precision by
changing both generation and composition objectives.

### 6.5 High-fidelity mammography validation

The full mammography run contained 840 fold-rate records, of which 686 were successful. At a target alert rate of 1%:

| Model | Realised alert rate | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Boosted paths | 1.03% | 0.904 | 0.400 | 0.551 |
| Iguanas | 1.03% | 0.853 | 0.378 | 0.522 |
| FIGS | 0.91% | 0.911 | 0.355 | 0.508 |
| SkopeRules | 0.91% | 0.832 | 0.327 | 0.466 |

At a 5% target, Iguanas spent essentially the full budget, while FIGS saturated near 1.32% realised alerts and retained approximately 0.831 precision. FIGS flags a smaller, purer population; Iguanas reaches a wider population and recovers more positives at lower precision. This is the operational distinction the benchmark is designed to expose.

### 6.6 Candidate-pool effects

Candidate-pool size materially affects composition. In the mammography equal-pool control, capping every method at eight candidates reduced boosted-path recall by approximately 0.295 and Iguanas recall by approximately 0.140 at the 5% target. Candidate count and fitted-tree count must therefore be reported alongside final quality metrics.

## 7. Failure Analysis and Limitations

Several baselines did not produce a usable disjunctive ruleset on every fold. RIPPER frequently produced no candidate satisfying the minimum quality conditions. BRL often produced an ordered list whose naive OR flattening flagged most of the population; it also failed the candidate-quality stage on many folds. These failures are retained explicitly rather than silently removed.

The broad 15-dataset comparison is a smoke configuration: one seed, three outer folds, and capped rows on some datasets. The Friedman report has complete model coverage for only three datasets and is exploratory, not definitive evidence of a global ranking.

The common disjunctive semantics are useful for comparing rule generators, but they are not native to every baseline. Ordered rule lists and unordered OR rulesets are different model classes. A future study should either compare each learner under native semantics or define separate native and disjunctive benchmark tracks.

The generation-grid ablation also has limitations. It uses one seed and three folds, and it measures one alert budget. The 10x4 configuration costs more computation, so its higher recall and F1 should not be interpreted as a free improvement. More seeds, budgets, and feature-selection methods are needed to determine when top-five targeting is preferable.

The experiment does not include a 6x1 arm, so it cannot isolate the marginal
contribution of the sample-weight axis from the class-weight axis. It also does
not report a formal diversity statistic such as pairwise disagreement, coverage
overlap, or unique-region count. Accordingly, "diversity control" describes the
generation mechanism and candidate-pool diagnostics here; it is not presented
as a causal diversity finding. No inferential claim is made from the current
cross-dataset means: the broad comparison uses one seed and three folds, and
the generation ablation uses 33 fold-level records per arm. Standard deviations,
multi-seed estimates, complete-case statistical tests, Friedman tests with
post-hoc comparisons, and critical-difference diagrams are required before
claiming statistically significant method rankings.

The paper does not claim that Iguanas is universally more precise than FIGS. The defensible claim is narrower: Iguanas makes candidate diversity, compute budget, composition objective, and alert volume explicit controls in one pipeline.

## 8. Reproducibility

The benchmark implementation is contained in `benchmarks/`. The protocol, dataset registry, preprocessing, model adapters, rule extraction, operating-point selection, reporting, and regression tests are versioned with the project. Principal artifacts are:

- `benchmarks/results/multi/smoke-20260910T191647Z/` for the 15-dataset baseline comparison;
- `benchmarks/results/weight-grid-all/results.csv` for the complete 11-dataset generation-grid experiment;
- `benchmarks/results/weight-grid-all/results-seeds012.csv` for the completed three-seed generation-grid experiment;
- `benchmarks/results/objective-ablation-selection.csv` for the controlled composition-objective experiment;
- `benchmarks/results/final-mammography/full-20260910T201334Z/` for the high-fidelity mammography validation.

The validated test command is:

```text
python3 -m pytest iguanas/tests benchmarks/tests -q -p no:cacheprovider --no-cov --ignore=iguanas/tests/test_onnx_converter.py
```

At the time of writing, this command passes 885 tests.

## 9. Conclusion

Iguanas should be understood as a framework for controlling the full path from loss steering to operational rule-set composition, not as a universally dominant classifier. Its class-weight and feature-derived sample-weight grids provide explicit diversity and compute controls. Its alert-budgeted composition makes the precision-recall-coverage trade-off explicit, and its exact branch-and-bound solver provides a principled reference against which greedy composition can be measured.

The experiments produce a precise division of strengths. FIGS is the strongest precision-first comparator and often produces a compact, high-purity ruleset. Iguanas is stronger when the objective values additional positive coverage and when the system must operate across a wider range of alert volumes. Within Iguanas, top-five feature targeting gives a small mean precision gain at a matched six-scale budget, while a ten-scale grid improves recall and F1 at substantially higher cost. Separately, precision-first composition raises mean precision from 0.613 to 0.766 but reduces mean recall from 0.465 to 0.272. Generation determines what trade-offs are available; composition chooses which feasible trade-off to deploy.

The practical conclusion is not that one algorithm wins universally. It is that rule-system evaluation must report the operating point, generation budget, and composition objective explicitly. Under a fixed alert budget, the meaningful question is whether a method can provide the required precision, recall, complexity, and compute trade-off. Iguanas makes those choices visible and controllable.

## Declarations

### Funding

To be completed by the authors at submission.

### Declaration of competing interest

To be completed by the authors at submission.

### CRediT authorship contribution statement

To be completed by the authors at submission.

### Data availability

The study uses publicly available datasets registered by name in the
benchmarking harness. Dataset acquisition and preprocessing are described in
the repository documentation. The generated result tables and benchmark
configuration are included in the repository artifacts listed above.

### Code availability

The Iguanas implementation, benchmark harness, tests, and manuscript source
are maintained in the open-source project repository [15]. The reproducibility
command and principal result paths are given above.

## References

1. Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence. 2019;1:206-215. https://doi.org/10.1038/s42256-019-0048-x.
2. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; 2016. p. 785-794. https://doi.org/10.1145/2939672.2939785.
3. Friedman JH, Popescu BE. Predictive learning via rule ensembles. Annals of Applied Statistics. 2008;2(3):916-954. https://doi.org/10.1214/07-AOAS148.
4. Letham B, Rudin C, McCormick TH, Madigan D. Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model. Annals of Applied Statistics. 2015;9(3):1350-1373. https://doi.org/10.1214/15-AOAS848.
5. Angelino E, Larus-Stone N, Alabi D, Seltzer M, Rudin C. Learning certifiably optimal rule lists. In: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; 2017. p. 35-44. https://doi.org/10.1145/3097983.3098047.
6. Lin J, Zhong C, Hu D, Rudin C, Seltzer M. Generalized and scalable optimal sparse decision trees. In: Proceedings of the 37th International Conference on Machine Learning; 2020. p. 6150-6160. PMLR 119.
7. Tan YS, Singh C, Nasseri K, Agarwal A, Duncan J, Ronen O, Epland M, Kornblith A, et al. Fast Interpretable Greedy-Tree Sums. Proceedings of the National Academy of Sciences. 2025;122(7):e2310151122. https://doi.org/10.1073/pnas.2310151122.
8. Cohen WW. Fast effective rule induction. In: Proceedings of the 12th International Conference on Machine Learning; 1995. p. 115-123. Morgan Kaufmann.
9. Lakkaraju H, Bach SH, Leskovec J. Interpretable decision sets: A joint framework for description and prediction. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; 2016. p. 1675-1684. https://doi.org/10.1145/2939672.2939874.
10. Wang F, Rudin C. Falling rule lists. In: Proceedings of the 18th International Conference on Artificial Intelligence and Statistics; 2015. PMLR 38:1013-1022.
11. Deng H. Interpreting tree ensembles with inTrees. International Journal of Data Science and Analytics. 2019;7:277-287. https://doi.org/10.1007/s41060-018-0144-8.
12. Bénard C, Biau G, da Veiga S, Scornet E. SIRUS: Stable and interpretable rule set for classification. Electronic Journal of Statistics. 2021;15:427-505. https://doi.org/10.1214/20-EJS1792.
13. Liu B, Mazumder R. FIRE: An optimization approach for fast interpretable rule extraction. In: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining; 2023. p. 1396-1405. https://doi.org/10.1145/3580305.3599353.
14. Rudin C, Chen C, Chen Z, Huang H, Semenova L, Zhong C. Interpretable machine learning: Fundamental principles and 10 grand challenges. Statistical Surveys. 2022;16:1-85. https://doi.org/10.1214/21-SS133.
15. PayPal. Iguanas: A Python library for rule generation and evaluation. GitHub repository. https://github.com/paypal/Iguanas/ (accessed 11 September 2026).
16. Dash S, Günlük O, Wei D. Boolean decision rules via column generation. In: Advances in Neural Information Processing Systems; 2018;31:7585-7594.
17. Wang T, Rudin C, Velez-Doshi F, Liu Y, Klampfl E, MacNeille P. Bayesian rule sets for interpretable classification. In: Proceedings of the 16th IEEE International Conference on Data Mining; 2016. p. 1269-1274. https://doi.org/10.1109/ICDM.2016.0171.
18. van Leeuwen M, Knobbe A. Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 2012;25:208-242. https://doi.org/10.1007/s10618-010-0213-9.

## Appendix: Result Integrity Notes

- All final test metrics are computed on data excluded from rule generation and selection.
- Generation is cached across alert-rate targets because it is independent of the target budget.
- Continuous preprocessing dtypes are preserved; one-hot columns alone are narrowed to compact integer types.
- Boolean emitters are not ranked by average precision.
- Failed folds are retained as explicit statuses rather than dropped before aggregation.
- The reported FIGS and Iguanas comparison distinguishes nominal target alert rate from realised alert rate.
