# Composing Rule Sets Under an Alert Budget

## Abstract

In fraud, abuse, compliance and credit-risk operations a detection model's output is not a score but a work queue: every flagged case consumes scarce analyst review capacity. A deployable rule system must therefore hit a specified alert budget, not merely rank positives well. We formalise budgeted rule-set composition as maximising positive coverage of a disjunction of conjunctive rules subject to a constraint on realised alert volume. Because both objective and constraint are coverage functions, the constraint is submodular rather than a linear knapsack, so the standard greedy guarantee for budgeted maximum coverage does not transfer — motivating exact search, for which we give a branch-and-bound composer with an admissible bound, implemented in the open-source Iguanas library. We evaluate six rule and reference models on 27 public tabular datasets over five seeds at *matched realised alert rate* rather than at a requested target, a distinction that reverses conclusions drawn from unmatched comparisons. Two findings emerge. First, budget attainment differs sharply: at a 5% budget FIGS saturates below the budget on 12 of 27 datasets and decision trees on 11, against 5 for the proposed composer. Second, at matched 5% capacity the composer attains the best mean recall (0.371), significantly exceeding RuleFit, SkopeRules and FIGS under Holm-corrected tests (p = 0.011, 0.022, 0.033; Friedman p = 0.033) while remaining indistinguishable from an unconstrained gradient-boosting reference (0.367, p = 0.890) — the only rule learner here to reach that ceiling. It concedes 11 precision points to FIGS, and does not hold at tighter budgets. A controlled ablation finds no detectable effect of generation-grid design, a ten-scale grid costing 67% more fitted trees for no measurable gain.

## Highlights

- Alert budget attainment, not ranking quality, determines rule-set deployability.
- The alert constraint is submodular, so greedy composition loses its guarantee.
- Matched realised alert rates reverse conclusions from unmatched comparisons.
- Budgeted composition matches a black-box ceiling on recall at equal alert cost.
- Generation-grid design shows no detectable quality effect at higher compute.

## Keywords

Rule-based classification; alert budget; interpretable machine learning; imbalanced classification; submodular optimisation; operational decision support

---

## 1. Introduction

A detection model deployed in a fraud, abuse, compliance or credit-risk operation does not produce decisions. It produces a queue. Each case it flags is routed to an analyst who must open it, gather context, decide, and document that decision. Review capacity is a staffed, budgeted quantity that changes on the timescale of hiring, not of model retraining. A model that flags twice as many cases as the operation can review has not doubled its coverage; it has created a backlog, and the cases that go unreviewed are chosen by queue order rather than by risk.

This makes the operational question different from the one conventional classification evaluation answers. The question is not how well a model ranks positives. It is: *given that we can review α% of the population, how many of the positives can we recover, with a rule set the policy owner will approve and the analyst can act on?*

Rule systems are attractive in this setting for reasons that are organisational as much as statistical. A rule is a concrete conjunction of conditions. It can be read in a review meeting, approved by a risk committee, versioned, attributed when it misfires, and evaluated in a production service with no model-serving infrastructure. Where a decision triggers human action against a customer, the distinction between an inherently interpretable model and a post-hoc explanation of a black box is consequential rather than aesthetic [1, 14].

Disjunctive rule sets — flag a case if any selected rule fires — are the dominant deployed form, and they interact with the alert budget in a specific way. Adding a rule to the disjunction can only increase both true and false positives. A small, high-precision rule set may be unable to spend the available budget at all: it *saturates*, leaving review capacity idle and positives unrecovered. A larger set spends more budget and recovers more positives at lower precision. Neither is universally right, and the choice is an operational one that depends on review cost and the cost of a missed positive.

Our starting observation is that this interaction is routinely obscured in published comparisons of rule learners, including our own earlier work. Methods are compared at a *requested* alert rate, but rule sets are discrete objects and frequently cannot realise the requested rate. In our benchmark, at a nominal 5% budget, the ratio between the highest and lowest realised alert rate across methods on the same dataset ranges from 1.20× to 13.38×, and exceeds 2× on 14 of 27 datasets. Comparing precision or recall across such points is comparing different operating conditions, and conclusions drawn from it are not sound. Section 6 shows that matching on *realised* alert rate materially changes the ranking of methods, and in the low-budget regime reverses it.

### 1.1 Contributions

1. **An operational evaluation protocol for boolean rule emitters.** Every method is placed on a common realised-alert-rate grid by interpolation within its feasible range, with explicit accounting for two distinct failure modes: *overshoot*, where a method cannot operate as conservatively as the budget requires, and *saturation*, where it cannot spend the budget at all (§5.3).
2. **A formulation of budgeted disjunctive composition**, with the observation that the alert constraint is submodular rather than a linear knapsack, so the classical $(1-1/e)$ guarantee for budgeted maximum coverage does not apply (§3.2). This motivates exact search as a principled choice rather than an engineering preference.
3. **An admissible bound and a best-first branch-and-bound composer** for precision, recall, accuracy and F-beta, with the submodularity argument that licenses reusing marginal gains at descendant nodes (§4.2).
4. **An empirical result on budget attainment**, which we argue is the primary practical differentiator between rule learners and which is not reported in the literature we surveyed (§6.1).
5. **A matched-budget comparison against an unconstrained gradient-boosting reference**, quantifying what interpretability costs at equal alert volume (§6.3).
6. **A negative result**: generation-grid design, the component of our own pipeline we had expected to matter most, has no statistically detectable effect on quality, while costing 67% more fitted trees in its expanded configuration (§6.4).

The composer, the bound and the generation pipeline are implemented in Iguanas, an open-source Python library for rule generation and evaluation [15]; it appears as `iguanas` in every results table below. We refer to it as *the proposed composer* in the text, because the contribution is the budgeted formulation and its solver rather than the software that implements them.

We are deliberately narrow about what the evidence supports. The recall advantage is established at a 5% budget on the 17 datasets where all methods are feasible, and does not reach significance at 2% or 1%, where that subset is smaller; and the proposed method concedes precision to FIGS throughout. The claims that hold across all budgets are about budget attainment and about the mechanism — saturation — that produces the recall differences.

---

## 2. Related work

**Disjunctive and Boolean rule set learning.** The model class studied here — an unordered disjunction of conjunctions, i.e. a DNF formula — has a direct literature. Dash, Günlük and Wei [16] learn DNF rule sets under an explicit complexity budget via integer programming with column generation, and Wang et al. [17] give a Bayesian formulation for unordered rule sets; Malioutov and Varshney [18] study LP relaxations for interpretable Boolean rules. These methods constrain rule-set *complexity*. The present work constrains realised *alert volume*, an orthogonal and, in capacity-constrained operations, binding constraint; the two can be imposed together, and the formulation in §3 admits a cardinality constraint alongside the budget.

**Ordered rule lists and certified optimisation.** Bayesian Rule Lists [4] produce ordered models; CORELS [5] and GOSDT [6] provide certified optimality for rule lists and sparse trees over a discretised space. Falling rule lists [10] order rules by decreasing risk. Ordered models define a nested family of operating points by truncation, which makes them natively budget-aware; we return to this in §5.2 and §8, as it is the correct way to compare them and a limitation of our current treatment.

**Rule ensembles and tree-ensemble extraction.** RuleFit [3] fits a sparse linear model over tree-derived rules. SkopeRules, inTrees [11] and SIRUS [12] extract and stabilise rules from tree ensembles; FIRE [13] optimises extraction directly; FIGS [7] grows a sum of shallow trees and is the strongest precision-oriented comparator in our benchmark. RIPPER [8] is the classical separate-and-conquer learner, and Interpretable Decision Sets [9] jointly optimise accuracy and interpretability for unordered sets.

**Subgroup discovery.** Our candidate shortlisting uses weighted relative accuracy (WRAcc), introduced for subgroup discovery in CN2-SD [19]. The broader paradigm of mining a large pool of candidate patterns and then selecting a diverse, non-redundant covering subset is subgroup set discovery [20]. Our generation-and-composition pipeline is structurally an instance of that paradigm, with the composition stage constrained by alert volume rather than by pattern-set diversity criteria; we regard the budget constraint and its optimisation consequences as the contribution, not the pipeline shape.

**Submodular optimisation.** Maximum coverage subject to a cardinality constraint is NP-hard, admits a $(1-1/e)$ greedy guarantee [21], and is inapproximable beyond that factor [22]. Our constraint is not a cardinality or linear-knapsack constraint but a submodular one; problems of this form are studied as submodular-cost submodular-knapsack by Iyer and Bilmes [23], who show that the standard guarantees degrade and give bicriteria results. §3.2 states the consequence for our setting.

**Budgeted and cost-sensitive detection.** Evaluation at a fixed review budget is standard practice in fraud detection and information retrieval (precision@k, recall@k), and example-dependent cost-sensitive learning [24] and work on credit-card fraud evaluation [25] address the asymmetric-cost setting. That literature largely concerns scoring models, where any budget is attainable by thresholding. The contribution here is that for *boolean* rule emitters, budget attainment is itself a non-trivial capability that varies by method — the central empirical finding of §6.1.

---

## 3. Problem formulation

### 3.1 Budgeted disjunctive composition

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}$ with $x_i \in \mathcal{X}$ and $y_i \in \{0,1\}$, and let $P = \{i : y_i = 1\}$ with $|P| = n_+$. A candidate rule $r_j$ is a conjunction of conditions on $\mathcal{X}$, identified with its coverage set $C_j = \{i : r_j(x_i) = 1\}$. Given a candidate pool $\mathcal{R} = \{r_1,\dots,r_M\}$, a selected rule set $S \subseteq \mathcal{R}$ predicts disjunctively:

$$\hat{y}_i(S) = \bigvee_{j \in S} r_j(x_i), \qquad C(S) = \bigcup_{j \in S} C_j .$$

Write the realised alert rate, true positives and false positives as

$$\mathrm{AR}(S) = \frac{|C(S)|}{n}, \qquad \mathrm{TP}(S) = |C(S) \cap P|, \qquad \mathrm{FP}(S) = |C(S) \setminus P| .$$

For an alert budget $\alpha \in (0,1]$ and a cardinality cap $k$, **budgeted disjunctive composition** is

$$\max_{S \subseteq \mathcal{R}} \ \mathrm{TP}(S) \quad \text{s.t.} \quad \mathrm{AR}(S) \le \alpha, \quad |S| \le k . \tag{1}$$

The objective counts recovered positives; the constraint caps the review workload. Precision-first and F-beta variants replace the objective while retaining the constraint, and are treated in §6.5 as an objective-sensitivity analysis.

### 3.2 The constraint is submodular, not a knapsack

$\mathrm{TP}$, $\mathrm{FP}$ and $|C(\cdot)|$ are all coverage functions, hence monotone and submodular. Problem (1) is therefore the maximisation of a monotone submodular objective subject to a **monotone submodular** constraint, since $\mathrm{AR}(S) \le \alpha$ is a sublevel set of a submodular function, not a linear budget $\sum_{j \in S} c_j \le \alpha$.

This distinction is consequential. For a linear (knapsack) budget, greedy selection by benefit-to-cost ratio achieves a constant-factor approximation, and for a cardinality constraint the classical $(1-1/e)$ bound holds [21]. Neither argument transfers here, because the "cost" of adding a rule depends on which rules are already selected: a rule that overlaps heavily with $C(S)$ consumes almost no additional budget, while the same rule added to a different partial solution may consume a great deal. Problems of this form belong to the submodular-cost submodular-knapsack family [23], for which the known results are bicriteria — guarantees are obtained only by relaxing the constraint, which in our setting means exceeding the alert budget and is operationally inadmissible.

We therefore do not claim an approximation guarantee for the greedy composer of §4.1, and we treat exact search (§4.2) as the principled reference rather than as an optional refinement. We state this as an observation about problem structure and its algorithmic consequence; we do not claim a new hardness result, and establishing the tight approximability of (1) is left open.

### 3.3 An admissible bound

Let $S$ be a partial solution with $s = k - |S|$ slots remaining and residual budget $\beta = \alpha n - |C(S)| \ge 0$. For a candidate $r_j \notin S$, write its marginal positive gain as $g_j(S) = \mathrm{TP}(S \cup \{r_j\}) - \mathrm{TP}(S)$, and let $g_{(1)}(S) \ge \dots \ge g_{(s)}(S)$ be the $s$ largest such gains. Define

$$\overline{\mathrm{TP}}(S) = \min\left(n_+, \ \mathrm{TP}(S) + \sum_{t=1}^{s} g_{(t)}(S), \ \mathrm{TP}(S) + \beta\right), \qquad \underline{\mathrm{FP}}(S) = \mathrm{FP}(S). \tag{2}$$

**Proposition 1 (admissibility).** For every $S' \supseteq S$ with $|S'| \le k$ and $\mathrm{AR}(S') \le \alpha$, we have $\mathrm{TP}(S') \le \overline{\mathrm{TP}}(S)$ and $\mathrm{FP}(S') \ge \underline{\mathrm{FP}}(S)$.

*Proof sketch.* Monotonicity of $\mathrm{FP}$ gives the second inequality. For the first, submodularity of $\mathrm{TP}$ implies $g_j(S'') \le g_j(S)$ for any $S'' \supseteq S$, so the gains available to any descendant are bounded by those computed at $S$; at most $s$ rules may still be added, giving the second term. The third term holds because each additional covered positive consumes at least one unit of residual alert budget. The first term is the total positive mass. $\square$

The third term in (2) is the budget-tightened bound and is frequently the binding one at tight budgets; it is what makes the alert constraint prune rather than merely restrict feasibility.

**Corollary 1.** Precision, recall, accuracy and $F_\beta$ are non-decreasing in TP and non-increasing in FP. Evaluating any such metric at $(\overline{\mathrm{TP}}(S), \underline{\mathrm{FP}}(S))$ therefore yields an upper bound on its value over all feasible descendants of $S$, and is admissible for branch-and-bound.

Two scope conditions matter for honest reporting. First, the bound is admissible with respect to the **retained candidate pool** $\mathcal{R}$; pool construction (§4.3) applies quality thresholds and correlation filtering and is heuristic, so the solver is exact over $\mathcal{R}$ and not over the space of all conjunctions. This is a weaker claim than the certified optimality of CORELS [5] or GOSDT [6], and we do not use the word "certified". Second, exactness additionally requires that no minimum-improvement heuristic be enabled.

---

## 4. Method

The pipeline has two stages, both implemented in Iguanas [15]. We emphasise at the outset that our evidence (§6.4) indicates the second stage is where the operationally relevant behaviour is determined; the first stage is reported for completeness and as a negative result.

### 4.1 Greedy composition

The default composer adds, at each step, the feasible candidate with the largest marginal positive gain, where a candidate is infeasible if its union with the current set would exceed $\alpha$. Because $C(\cdot)$ is monotone, an over-budget set cannot be made feasible by further additions, so infeasible branches are terminal. This is the natural greedy heuristic for (1); per §3.2 it carries no approximation guarantee under a submodular constraint.

### 4.2 Exact branch-and-bound composition

The exact composer explores subsets of $\mathcal{R}$ in best-first order by the bound of Corollary 1. A node is pruned when its bound does not exceed the incumbent objective, or when its realised alert volume already exceeds $\alpha$. Ties are broken by smaller cardinality, so that among equal-objective solutions the composer returns the more compact rule set. Nodes expanded, candidate sets evaluated and metric evaluations are recorded per run and reported in §6.7.

### 4.3 Candidate generation and pool construction

Candidates are root-to-leaf paths extracted from gradient-boosted trees [2], pooled across a grid of configurations. Two weighting mechanisms vary the fitted trees: `scale_pos_weight`, which alters the relative cost of minority-class errors, and sample weights derived from a feature transformation, which alter which observations dominate the fitted splits. Scale values are logarithmically spaced from 1 to $\max(2, n_-/n_+)$ computed on the generation split, the floor ensuring a non-degenerate grid on mildly imbalanced data. A grid of six scales by four weight schedules is written 6×4 and yields 24 configurations; at 15 trees per configuration this is a nominal 360 fitted trees.

Sample-weight schedules are increasing $((1+x)^p,\ \log(1+x))$ and decreasing $(1/(1+x)^p$, inverse-log$)$ families with $p \in \{0.25, 0.5, 1, 2, 4\}$, applied to the absolute value of the chosen feature. The feature is chosen either as the highest-variance feature (*variance* mode) or from the five most important features of a class-balanced importance model fitted on the generation split (*top-five* mode). No test data enters this choice.

Pooled candidates are deduplicated, filtered to minimum precision and recall, shortlisted by WRAcc [19],

$$\mathrm{WRAcc}(r) = \frac{|C_r|}{n}\left(\frac{|C_r \cap P|}{|C_r|} - \frac{n_+}{n}\right), \qquad |C_r| > 0,$$

and reduced by removing highly correlated candidates so that near-duplicate paths are not mistaken for pool diversity.

---

## 5. Experimental protocol

### 5.1 Splits, and what is and is not disjoint

Each outer fold partitions the data into a development portion and a test portion. Within development, the harness supports three selection regimes: `train`, where generation and selection both use the whole development portion; `holdout`, where a selection split is carved out of development; and `oracle`, where selection uses the test rows and which exists only to measure how much the selection stage can overfit.

**The runs reported here use `train`.** Generation and selection therefore operate on the same rows, and the two are *not* disjoint. Test rows never enter generation or selection: a runtime leak guard raises if the generation and test index sets intersect, and under `train` it additionally checks selection against test. Reported test metrics are consequently unbiased, but selection-stage metrics are optimistic and we do not use them for headline claims.

We quantify that optimism rather than asserting it is small. In the `train` arm, mean selection-split F1 exceeds mean test F1 by 0.038 (0.5029 vs 0.4645), pooled over all models on `mammography` at one seed; for the proposed composer alone the gap is 0.034 (0.5196 vs 0.4859). In the `oracle` arm the two coincide by construction, which is what makes it an upper bound rather than an estimate. All results below are test-set results.

Preprocessing is fitted on development data only. Candidate generation is cached across alert-rate targets because the pool does not depend on $\alpha$.

### 5.2 Models and semantics

We evaluate the proposed composer (`iguanas`) against FIGS [7], SkopeRules, RuleFit [3], a single decision tree, and `gbm_ceiling`, an unconstrained gradient-boosting classifier thresholded to the target alert rate. The last is not interpretable and is not a competitor; it is a reference for what the alert budget permits at all, and the gap to it is our measure of what the rule constraint costs (§6.3).

RIPPER [8] and Bayesian Rule Lists [4] are reported separately. Both are ordered models, and flattening an ordered list into a disjunction changes its semantics; in addition both frequently failed to yield a usable rule set under our candidate-quality conditions (§6.8). We regard our treatment of them as a limitation rather than a result (§8): the correct comparison truncates the ordered list to the budget, which preserves native semantics, and we have not yet implemented it.

All models receive the same preprocessing, the same folds and seeds, and the same alert-rate grid. Hyperparameters for all baselines are library defaults; the proposed method's generation grid is the 6×4 variance configuration. This asymmetry — a 24-configuration generation grid against default baselines — favours the proposed method on candidate-pool size, which §6.6 shows is consequential. We report it explicitly because it qualifies the comparison, and because the precision results run against the direction of that advantage.

### 5.3 Matched realised alert rate

Requested and realised alert rates differ, often substantially. For each dataset and model we average realised alert rate and test metrics over seeds and folds at each requested target, producing an operating curve, and then evaluate each model at a common grid of realised rates $\{1\%, 2\%, 5\%\}$ by linear interpolation **within the model's realised range only**. Outside that range we record one of two states rather than extrapolating:

- **overshoot** — the model's smallest realised alert rate exceeds the budget, so it cannot operate that conservatively. Operationally this is a budget violation.
- **saturation** — the model's largest realised alert rate falls below the budget, so it cannot spend the available capacity. Its ceiling metrics are carried forward, which is the correct operational reading: this is the best that model can do given that capacity.

Because a model's feasible set of datasets depends on the budget, means taken over all datasets average different subsets for different models and are not directly comparable. Our **primary comparison is therefore the common subset** on which every model is feasible, with $n$ stated; full-sample means and feasibility counts are reported alongside.

### 5.4 Metrics and testing

We report precision, recall, F1, realised alert rate, condition count, fitted-tree count, candidate-rule count, and fit and composition time. Significance across datasets uses the Friedman omnibus test on per-dataset means at matched budget, following Demšar [26]. Because our hypothesis compares one control method against the others rather than all pairs, the post-hoc is Holm-corrected pairwise Wilcoxon signed-rank, which Demšar recommends over Nemenyi in the control-comparison setting; we report Nemenyi alongside it for completeness. The unit of analysis is the dataset, so the five seeds reduce per-cell noise but do not increase test power. Common-subset sizes are $n = 3$ at a 1% budget, $n = 8$ at 2% and $n = 17$ at 5%, and we report non-significant results as such rather than interpreting rank orders as established differences. Average precision is recorded but not used to rank boolean emitters, which occupy a single operating point.

### 5.5 Datasets and configuration

Twenty-seven public tabular datasets, five seeds, three outer folds, six alert-rate targets $\{0.5\%, 1\%, 2\%, 3\%, 5\%, 10\%\}$, cardinality cap $k = 8$, 120 s fit timeout. Rows are stratified-subsampled to 4000 with a minimum-positive floor, which is why the most imbalanced datasets retain more rows. Positive rates span 0.17% to 39.4%, so the benchmark covers both the extreme-imbalance regime in which alert budgets bind hardest and the near-balanced regime in which they do not.

**Table 1. Datasets (rows as used after subsampling).**

| Dataset | Rows | Features | Positive % | Imbalance | Domain |
|---|---:|---:|---:|---:|---|
| creditcard | 115775 | 29 | 0.17 | 577.9 | fraud |
| pc2 | 5589 | 36 | 0.41 | 242.0 | software defect |
| mc1 | 9466 | 38 | 0.72 | 138.2 | software defect |
| satellite_anomaly | 5100 | 36 | 1.47 | 67.0 | anomaly |
| aps_failure | 11055 | 170 | 1.81 | 54.2 | industrial |
| mammography | 8602 | 6 | 2.33 | 42.0 | medical |
| wilt | 4005 | 5 | 5.39 | 17.5 | remote sensing |
| sick | 3772 | 29 | 6.12 | 15.3 | medical |
| ozone_level_8hr | 2534 | 72 | 6.31 | 14.8 | environmental |
| pc1 | 1109 | 21 | 6.94 | 13.4 | software defect |
| pc3 | 1563 | 37 | 10.24 | 8.8 | software defect |
| bank_marketing | 4001 | 16 | 11.70 | 7.5 | marketing |
| pc4 | 1458 | 37 | 12.21 | 7.2 | software defect |
| churn | 4003 | 20 | 14.14 | 6.1 | churn |
| thoracic_surgery | 470 | 16 | 14.89 | 5.7 | medical |
| kc1 | 2109 | 21 | 15.46 | 5.5 | software defect |
| speeddating | 4001 | 120 | 16.47 | 5.1 | social |
| jm1 | 4000 | 21 | 19.35 | 4.2 | software defect |
| hepatitis | 155 | 19 | 20.65 | 3.8 | medical |
| credit_default | 4001 | 23 | 22.12 | 3.5 | credit default |
| blood_transfusion | 748 | 4 | 23.80 | 3.2 | medical |
| adult | 3999 | 14 | 23.93 | 3.2 | census |
| phoneme | 4000 | 5 | 29.35 | 2.4 | speech |
| credit_g | 1000 | 20 | 30.00 | 2.3 | credit default |
| diabetes | 768 | 8 | 34.90 | 1.9 | medical |
| ionosphere | 351 | 34 | 35.90 | 1.8 | signal |
| spambase | 4000 | 57 | 39.40 | 1.5 | spam |

Seven of the twenty-seven are NASA MDP software-defect datasets, a family with documented quality problems [27]; §6.2 reports a sensitivity analysis excluding them. At 26% of the benchmark they no longer dominate it, which was a weakness of an earlier thirteen-dataset version of this study. The 5% result loses significance when they are removed; §6.2 examines this and attributes it primarily to the reduced sample size.

---

## 6. Results

### 6.1 Budget attainment is the primary differentiator

Before comparing quality, we ask whether each method can hit the budget at all. Table 2 reports the mean of each model's maximum realised alert rate, and the number of datasets on which it cannot reach a 5% budget.

**Table 2. Budget reach (27 datasets, 5 seeds).**

| Model | Mean max realised AR | Cannot reach 5% |
|---|---:|---:|
| gbm_ceiling (reference) | 8.20% | 6 / 27 |
| **iguanas** | **8.05%** | **5 / 27** |
| RuleFit | 7.69%† | 6 / 26 |
| SkopeRules | 5.93% | 10 / 27 |
| Decision tree | 5.73% | 11 / 27 |
| FIGS | 5.23% | 12 / 27 |

† RuleFit exceeds the 120 s per-fit budget on every fold of `creditcard` (115,775 rows), the largest dataset in the benchmark, and so yields no operating curve there; its mean is over the remaining 26 datasets and is not strictly comparable with the other rows. This is a recorded timeout rather than missing data — see §6.8.

The spread is large and operationally consequential. FIGS cannot spend a 5% review budget on 12 of 27 datasets and the decision tree on 11; the proposed composer fails on 5 and comes within 0.15 percentage points of the unconstrained gradient-boosting reference (8.049% vs 8.201%, a gap of 0.152 pp). Table 3 shows the complementary failure at the tight end.

**Table 3. Budget feasibility (27 datasets).** *Overshoot* = the model's smallest realised alert rate exceeds the budget, so it cannot operate this conservatively. *Saturated* = its largest realised alert rate falls below the budget, so it cannot spend the capacity. *Hits* counts every dataset that is not an overshoot, and therefore **includes** the saturated cases, which are scored at their ceiling per §5.3; the number of datasets genuinely interpolated at the stated budget is hits minus saturated.

| Model | 1%: hits / over / sat | 2%: hits / over / sat | 5%: hits / over / sat |
|---|---:|---:|---:|
| **iguanas** | **14 / 13 / 3** | **22 / 5 / 3** | **26 / 1 / 5** |
| gbm_ceiling | 11 / 16 / 2 | 20 / 7 / 3 | 25 / 2 / 6 |
| FIGS | 11 / 16 / 3 | 16 / 11 / 6 | 23 / 4 / 12 |
| SkopeRules | 9 / 18 / 3 | 14 / 13 / 5 | 26 / 1 / 10 |
| Decision tree | 9 / 18 / 1 | 14 / 13 / 3 | 25 / 2 / 11 |
| RuleFit | 5 / 21 / 0 | 8 / 18 / 4 | 18 / 8 / 6 |

At a 1% budget RuleFit can operate on only 5 of 26 datasets and the decision tree and SkopeRules on 9 of 27; the proposed composer manages 14, of which 11 are genuine operating points and 3 are saturation ceilings. Budget attainment, in both directions, is where the methods differ most clearly, and it is not reported in the comparisons we surveyed.

### 6.2 Quality at matched realised alert rate

Table 4 gives the primary comparison: the common subset of datasets on which all six models are feasible.

**Table 4. Recall and precision at matched realised alert rate, common feasible subset, 5 seeds.** Some entries are saturation ceilings carried forward per §5.3 rather than operating points at the stated budget. At 5% (n = 17) the number of such entries per model is: FIGS 10, decision tree 9, SkopeRules 9, RuleFit 6, gbm_ceiling 5, iguanas 4; at 2% (n = 8): FIGS 5, RuleFit 4, SkopeRules 4, decision tree 2, gbm_ceiling 2, iguanas 2. The comparison is therefore matched on *available capacity*, not on realised alert rate for every cell, and models that saturate are credited with the best they can achieve under that capacity.

| Budget | n | Model | Recall | SD | Precision |
|---|---:|---|---:|---:|---:|
| 2% | 8 | gbm_ceiling | 0.375 | 0.244 | 0.549 |
| | | **iguanas** | **0.371** | 0.237 | 0.554 |
| | | SkopeRules | 0.333 | 0.246 | 0.608 |
| | | RuleFit | 0.331 | 0.210 | 0.550 |
| | | FIGS | 0.321 | 0.210 | **0.689** |
| | | Decision tree | 0.320 | 0.148 | 0.486 |
| 5% | 17 | **iguanas** | **0.371** | 0.273 | 0.496 |
| | | gbm_ceiling | 0.367 | 0.274 | 0.487 |
| | | Decision tree | 0.327 | 0.227 | 0.490 |
| | | RuleFit | 0.310 | 0.224 | 0.499 |
| | | SkopeRules | 0.309 | 0.231 | 0.554 |
| | | FIGS | 0.303 | 0.229 | **0.607** |

At a matched 5% capacity the proposed composer has the highest mean recall and the best mean Friedman rank (2.53), ahead of the unconstrained reference (2.71), the decision tree and SkopeRules (3.76 each), FIGS (4.00) and RuleFit (4.24). **The Friedman test is significant** ($\chi^2 = 12.16$, $p = 0.033$, $n = 17$, $k = 6$).

Because the hypothesis concerns one control method against the others rather than all pairs, we follow Demšar [26] in using Holm-corrected pairwise Wilcoxon signed-rank tests rather than the Nemenyi post-hoc, which is known to be conservative in the control-comparison setting. At 5% the proposed composer has significantly higher recall than RuleFit (Holm $p = 0.011$, mean difference $+0.062$), SkopeRules ($p = 0.022$, $+0.062$) and FIGS ($p = 0.033$, $+0.068$). It is **not** significantly different from the decision tree ($p = 0.319$, $+0.044$), nor from the unconstrained gradient-boosting reference ($p = 0.890$, $+0.004$) — the latter being the intended result, since parity with the ceiling is the claim. The Nemenyi post-hoc separates no pair at this sample size (smallest of all pairwise $p$-values: iguanas–RuleFit, $p = 0.084$), which we report for completeness.

At 2% the Friedman test is not significant ($\chi^2 = 9.93$, $p = 0.077$, $n = 8$) and no pairwise comparison survives correction. Nor is the ordering identical: by mean recall the gradient-boosting reference is first at 2% (0.375) and the proposed composer second (0.371), a reversal of the 5% result, and the decision tree moves from third to last. Only the top two by mean rank are preserved (2.25 and 2.38). At 1% the common subset is three datasets and no comparison is meaningful — itself a consequence of the overshoot behaviour in Table 3.

FIGS retains a substantial precision advantage (0.607 vs 0.496 at 5%; 0.689 vs 0.554 at 2%) at significantly lower recall (Holm $p = 0.033$ at 5%). The precision difference itself is large but does not reach significance under the same correction (raw Wilcoxon $p = 0.017$ at 5%, Holm $p = 0.087$), so we describe it as a consistent tendency rather than an established difference. The advantage is large on average rather than universal: FIGS is the more precise of the two on 12 of 17 datasets at 5%, and less precise on `ionosphere`, `kc1`, `pc3`, `pc4` and `thoracic_surgery`. This is the operational trade-off the protocol is designed to expose, and a genuine strength of FIGS rather than an artefact.

**Sensitivity to the defect-dataset family.** Six of the seven NASA MDP datasets are present in the 5% common subset; excluding them reduces it to $n = 11$, and on that subset the Friedman test is not significant ($\chi^2 = 8.04$, $p = 0.154$). The effects themselves are largely stable. Table 4b gives the paired comparisons on both samples.

**Table 4b. Iguanas versus each comparator at a matched 5% budget, full sample and excluding NASA MDP.** $d_z$ is the matched-pairs effect size.

| Comparator | Full: mean diff | $d_z$ | wins | ex-NASA: mean diff | $d_z$ | wins |
|---|---:|---:|---:|---:|---:|---:|
| FIGS | +0.068 | 0.70 | 12/17 | **+0.070** | 0.60 | 6/11 |
| SkopeRules | +0.062 | 0.79 | 12/17 | +0.060 | 0.68 | 7/11 |
| RuleFit | +0.062 | 0.57 | 15/17 | **+0.080** | 0.65 | **11/11** |
| Decision tree | +0.044 | 0.34 | 11/17 | **+0.083** | 0.70 | 7/11 |
| gbm_ceiling | +0.004 | 0.13 | 9/17 | +0.004 | 0.14 | 6/11 |

Four of the five mean differences are preserved or larger on the reduced sample, and the fifth is essentially unchanged (SkopeRules, 0.062 → 0.060). Effect sizes move in both directions — they fall for FIGS (0.70 → 0.60) and SkopeRules (0.79 → 0.68) and rise for RuleFit and the decision tree — but all four remain in the 0.60–0.70 band, and parity with the gradient-boosting reference is unchanged (+0.004 on both samples). Against RuleFit the proposed composer wins on all 11 remaining datasets. A bootstrap at the observed ex-NASA effect gives power 0.53 at $n = 11$, rising to 0.89 at $n = 17$, so non-significance is the expected outcome at this sample size rather than evidence of a vanishing effect. We therefore do not think the defect-dataset family is driving the result, but we cannot demonstrate that with the datasets available: doing so requires roughly six to nine additional non-defect imbalanced datasets, not a different analysis of the present ones.

**Robustness to near-balanced datasets.** An alert budget is only a binding operational constraint when positives are scarce; at a 39% base rate, a 5% review capacity is not the quantity that limits a deployment. Restricting to the 15 datasets in the common subset with a positive rate below 20% — excluding `blood_transfusion` and `ionosphere` — therefore tests the claim on the population the paper is about. The result strengthens: $\chi^2 = 13.71$, $p = 0.018$, with the proposed composer first by mean rank (2.27) and Holm-corrected advantages over SkopeRules ($p = 0.010$), RuleFit ($p = 0.017$) and FIGS ($p = 0.017$), and continued parity with the reference ($p = 0.847$). We report the unrestricted common subset ($n = 17$) as the primary analysis regardless, because it is the more inclusive and more conservative choice; the restriction is offered as a robustness check and was not used to select the headline result. Across every threshold we examined (no restriction, 30%, 25%, 20%, 15%, 10%) the omnibus $p$ lies between 0.018 and 0.033 and the proposed composer ranks first, so the conclusion is not an artefact of where the line is drawn.

**A note on how this sample size was reached.** An earlier version of this study used 13 datasets and reported $p = 0.098$ at 5%. Rather than search over tests or comparator subsets, we expanded the benchmark to every verified dataset in the harness registry, fixing the primary comparison (recall at matched 5% capacity, control versus each baseline, Holm-corrected) before running the additional datasets. A bootstrap power analysis on the 13-dataset result estimated power of 0.59 at $n = 11$ and 0.95 at $n = 20$, consistent with the significant result now obtained at $n = 17$. We note that reducing the comparator set would have *lowered* power, and we did not do it.

### 6.3 What the rule constraint costs

Table 4 permits a direct reading of the price of interpretability at equal available capacity. On the same common subsets as Table 4, mean recall relative to the unconstrained gradient-boosting reference is:

| Model | 2% (n = 8) | 5% (n = 17) |
|---|---:|---:|
| **iguanas** | **−0.004** | **+0.004** |
| Decision tree | −0.055 | −0.040 |
| RuleFit | −0.044 | −0.058 |
| SkopeRules | −0.043 | −0.058 |
| FIGS | −0.054 | −0.065 |

We restrict this table to the 2% and 5% common subsets; at 1% the common subset is too small (§6.2) and is not reported.

The proposed composer tracks the black-box ceiling to within 0.004 recall at both budgets — a difference that is not statistically significant at 5% (Holm $p = 0.890$) — and is the only method that does not fall measurably behind it, while the other rule learners are 4.3–5.5 recall points behind at 2% and 4.0–6.5 points behind at 5%. For a capacity-constrained operation this is the central practical claim of the paper: at the alert volumes at which such systems are actually run, an auditable disjunction of rules need not cost measurable recall relative to an unconstrained gradient-boosting model. Two qualifications apply. The comparison is on recall at matched capacity and does not imply parity on ranking metrics, where a scoring model retains the advantage of a full ordering. And part of the gap for the saturating methods reflects their inability to spend the budget (Table 4 caption) rather than weaker rules per alert — which is the point of §6.1, but means the number should be read as the cost of the *deployed system*, not of the rule language.

### 6.4 Generation-grid design: a negative result

We had expected candidate-generation design to be the principal lever. It is not. Table 5 compares three generation grids over 3 seeds and 3 folds (99 fold-level runs per arm; 33 paired dataset×seed units), holding composition fixed.

This ablation uses 11 of the 27 datasets in Table 1 and 3 seeds rather than 5, so it has lower power than the main comparison and its cells are not directly comparable with Table 4.

**Table 5. Generation-grid ablation (11 datasets, 3 seeds, 99 runs per arm). Trees and rules are measured means, not nominal.**

| Grid | Trees | Rules | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Variance feature, 6×4 | 346.4 | 174.6 | 0.611 ± 0.202 | 0.456 ± 0.254 | 0.480 ± 0.210 |
| Top-five features, 6×4 | 359.1 | 196.7 | 0.620 ± 0.203 | 0.460 ± 0.261 | 0.486 ± 0.218 |
| Top-five features, 10×4 | 598.5 | 288.0 | 0.616 ± 0.191 | 0.461 ± 0.258 | 0.488 ± 0.213 |

No difference is statistically detectable. Friedman across the three arms gives $p = 0.052$ for precision, $p = 0.390$ for recall and $p = 0.365$ for F1. Pairwise Wilcoxon comparisons of the two 6×4 arms give $p = 0.790$ for precision (mean difference $+0.009$, 95% CI $[-0.009, +0.027]$), $p = 0.304$ for recall and $p = 0.292$ for F1; all intervals straddle zero. Holding feature selection fixed at top-five and expanding the scale grid from six to ten values changes F1 by $+0.001$ ($p = 0.501$) while fitting **67% more trees** (598.5 vs 359.1) and generating 46% more candidate rules (288.0 vs 196.7).

The honest conclusion is that within the range examined, generation-grid design does not measurably affect deployed quality, and the expanded grid is a pure compute cost. We note this supersedes a claim of a precision improvement that we drew from a single-seed version of the same experiment; the apparent effect did not survive replication at three seeds. Because both mechanisms were varied together in all arms, we also cannot attribute the pool's usefulness to class weighting versus sample weighting separately; a factorial isolating the two axes is required and is not reported here (§8).

### 6.5 Composition-objective sensitivity

Replacing the coverage-first objective with a precision-first objective on the same candidate pool raises mean precision from 0.613 to 0.766 and lowers mean recall from 0.465 to 0.272. The two arms realise mean alert rates of 3.58% and 1.48% respectively. **They therefore do not share an operating point**, and per §5.3 this is an objective-sensitivity analysis, not a matched comparison: a precision-first objective selects a smaller rule set that spends less budget, and part of the precision difference is attributable to the lower alert rate rather than to the objective. A matched-budget version of this ablation, constraining both arms to equal realised alert rate, is required to separate the two effects and is not yet available.

### 6.6 Candidate-pool size is a benchmark confound

Capping every method at eight candidate rules on `mammography` at a 5% *requested* target reduced the gradient-boosting reference's recall by 0.295 (0.670 → 0.375) and the proposed composer's by 0.140 (0.590 → 0.450). These drops are large, but they are not matched on realised alert rate: capping also collapsed realised alert rate from 3.43% to 1.17% and from 3.19% to 1.45% respectively. By our own criterion (§5.3) this is an unmatched comparison, and it therefore does not establish that pool size affects quality more than the method differences in Table 4; the recall loss and the alert-rate loss are confounded.

What the result does establish is that candidate-pool size is not a neutral implementation detail, and that methods differ in it by more than an order of magnitude at equal settings — 147.0 candidate rules for the proposed generator against 6.1 for FIGS and 15.6 for SkopeRules (Table 6). A comparison that does not report candidate count therefore leaves a first-order factor unreported, including our own §6.2, where the proposed method's larger pool is an acknowledged advantage (§5.2). A matched-alert-rate version of the pool-cap experiment is needed to quantify the effect and is listed in §8.

### 6.7 Exact versus greedy composition

On `mammography`, exact branch-and-bound composition attained selection F1 0.621 against 0.547 for greedy composition under the same requested budget, a gap of 0.074. Three caveats are needed, and together they mean this is an existence proof rather than a characterisation. The figure is on the selection split (§5.1), where we measured 0.038 F1 of optimism. It is one dataset. And the two arms do not share a realised operating point: exact composition realised 1.74% selection alert rate against greedy's 3.19%, so the exact arm achieved higher F1 while spending roughly half the alert volume. That direction is favourable — it is a better operating point on both axes — but it is not a matched comparison, and the F1 gap cannot be attributed to search quality alone.

The alert constraint does contribute materially to pruning through the third term of (2). A systematic study — solve time and nodes expanded as a function of pool size, timeout rates, and the distribution of the greedy optimality gap across datasets at matched realised alert rate — is the most important missing piece of evidence for the exact composer and is not included here (§8).

### 6.8 Complexity, compute and availability

**Table 6. Complexity and compute at a 5% budget (27 datasets, 5 seeds, 405 cells per model; RuleFit 387 and SkopeRules 395 owing to failures).**

| Model | Conditions | Fit (s) | Compose (s) | Trees | Candidate rules |
|---|---:|---:|---:|---:|---:|
| RuleFit | 4.52 ± 3.03 | 4.49 | 0.00 | — | 9.5 |
| SkopeRules | 6.02 ± 4.85 | 0.43 | 0.00 | — | 15.6 |
| FIGS | 8.17 ± 6.28 | 0.45 | 0.00 | — | 6.1 |
| **iguanas** | 9.59 ± 7.00 | 1.10 | 0.04 | 344 | 147.0 |
| Decision tree | 9.77 ± 7.09 | 1.00 | 0.00 | 12 | 11.9 |
| gbm_ceiling | 12.90 ± 10.06 | 0.45 | 0.16 | 200 | 1412.9 |

The proposed composer's recall is not free. It maintains about as many conditions as a decision tree and 17% more than FIGS, and it fits 344 trees per fold against 12 for the decision tree and none for FIGS, RuleFit and SkopeRules, which do not fit boosted ensembles at all — a roughly 29-fold larger fitted-model count than the decision tree, though still around one second per fold at these dataset sizes. RuleFit is the most compact and by far the slowest to fit.

**Baseline availability.** Over the 27-dataset, five-seed run (2430 fold×budget cells per model), the proportion of cells yielding no usable rule set was: RIPPER 77.5% (1884 of 2430), SkopeRules 2.5% (60), RuleFit 0.8% (18 of 2340), and 0% for the decision tree, FIGS, `gbm_ceiling` and the proposed composer. RIPPER's and SkopeRules' failures, and RuleFit's 18 on `aps_failure`, are cases in which no candidate rule satisfied the minimum quality conditions.

RuleFit fails differently on `creditcard`, and the distinction matters for a deployment-oriented comparison. It does not produce a poor rule set there; it produces none at all, because fitting exceeds the 120 s budget on every fold — recorded explicitly as `timeout` with the message *exceeded the 120s fit budget* in the single-seed run, where all 18 of its `creditcard` cells carry that status. This is consistent with Table 6, where RuleFit is the slowest method to fit by a factor of four (4.49 s mean against 1.10 s for the proposed composer) while producing the most compact rule sets. Counting `creditcard` as unavailability rather than excluding it raises RuleFit's overall rate from 0.8% to 4.4%. We report the timeout rather than raising the fit budget, since a larger budget would change every other result and would reward a method precisely where it is least deployable: on the largest dataset in the benchmark, at 115,775 rows.

In the earlier single-seed run that also included BRL, BRL was unusable on 46.7% of cells. RIPPER and BRL are excluded from Tables 2–4, as their availability is too low to support a matched-budget comparison. CORELS could not be constructed in our environment, as the installed `imodels` version exposes none of its expected entry points; it is cited in §2 but not benchmarked, which is a gap. We retain failures as explicit statuses rather than dropping them before aggregation, and we exclude RIPPER and BRL from Tables 2–4 because their availability is too low to support a matched-budget comparison.

### 6.9 Generation backend choice: tree-based versus classical rule induction

The results above compare *composition* strategies at a fixed generation backend. A separate question is whether the choice to generate candidates from tree ensembles at all — as opposed to a classical, non-tree rule-induction algorithm — is itself supported by evidence, since we do not claim a novel rule-induction algorithm (§1.1). This subsection reports a supplementary analysis addressing that question. It uses a different, simpler protocol than §5–§6.8: a single 70/30 train/test split per seed (five seeds), full-coverage greedy composition with no alert-budget matching, and 24 of the 27 Table 1 datasets — `pc2` is outside this harness's dataset registry, and `aps_failure`/`creditcard` are excluded on cost grounds. Because it is not budget-matched, it is not directly comparable to Tables 2–4, and we report it as a separate, narrower claim about generation rather than as an extension of the main results.

We compare a *tree-like* family — the XGBoost backend used elsewhere in this paper, plus LightGBM and Random Forest (Gini and entropy criteria) backends added to Iguanas since the main experiments, and RuleFit, which derives its candidates from a fitted tree ensemble — against a *non-tree* family: RIPPER [8], Bayesian Rule Lists [4], SLIPPER [28], OneR [29], and a single-feature-at-a-time sequential decision list (`GreedyRuleList`) grown by the same impurity-reduction criterion as the reference decision tree. Under this protocol's simpler, unmatched full-coverage criterion, RIPPER produces a rule set on all 120 dataset×seed cells (0% failure), in contrast to the 77.5% failure rate reported for it in §6.8 under the matched-realised-alert-rate protocol; we take this as evidence that RIPPER's earlier failures there were driven mainly by an inability to hit a narrow low-alert-rate target rather than by an inability to find any rule at all. RuleFit fails on 1 cell (0.8%), OneR on 33 (27.5%), `GreedyRuleList` on 39 (32.5%) and SLIPPER on 69 (57.5%), the last mostly from its boosting loop failing to find a base rule better than random on a given re-weighted sample rather than from a lack of qualifying candidates. BRL's own failure rate is 75% (90 of 120) here — higher, not lower, than the 46.7% reported in §6.8 — because keeping its candidate-mining tractable at all on class-balanced datasets required tightening its support and cardinality parameters (§8), which both prevents the multi-minute-per-fold blowups we observed at the more permissive default settings and prunes away more of its candidate itemsets before it ever reaches the rule-list search.

Paired by dataset ($n=24$, one mean MCC per family per dataset, averaged over whichever family members produced a result), the tree-like family exceeds the non-tree family on 23 of 24 datasets, with a mean advantage of 0.171 MCC. A one-sided Wilcoxon signed-rank test rejects equality ($p = 1.8\times10^{-7}$); a paired $t$-test agrees ($t=6.88$, $p=1\times10^{-6}$). The single exception, `thoracic_surgery`, is a dataset on which every method scores near-zero MCC and the sign of the difference is not meaningful. A Friedman test restricted to the five tree-like models alone, where all five report a result on all 24 datasets, does **not** reject equality ($\chi^2=9.37$, $p=0.053$; mean ranks 2.58–3.79), consistent with the generation-grid finding of §6.4: once generation is tree-based, further choices among tree ensembles (boosting versus bagging, Gini versus entropy) have at most a marginal, not clearly detectable, effect.

We read this as supporting evidence for a design choice rather than a competing contribution. It does not show that tree ensembles are a better *rule representation* — RuleFit, the one member of the tree-like family that is not itself a tree, tracks the ensemble backends closely and sits well ahead of every non-tree method — so much as that extracting rules from a strong ensemble learner outperforms inducing them directly with the classical algorithms available to us, at least at the operating point this protocol measures (full-coverage MCC, not a matched alert budget). The comparison inherits three limitations from its non-tree baselines' availability: BRL and RIPPER's low success rates mean their contribution to the non-tree family mean is a selected, not a representative, subset of datasets, the same caveat §6.8 raises for Tables 2–4; SLIPPER's failures are a different mechanism (occasional convergence failures in its boosting loop, not lack of qualifying candidates); and we have not run this comparison under the matched-realised-alert-rate protocol of §5.3, which is the more defensible comparison and is left for future work (§8).

---

## 7. Discussion: choosing a method under a review budget

The results support a decision procedure rather than a winner.

**Identify the binding budget first.** If the operation can review 5% of the population, methods that saturate below 5% leave capacity idle; Table 2 shows this happens for FIGS on 12 of 27 datasets and for decision trees on 11. If the operation can review 1%, the risk inverts: most methods overshoot — RuleFit on 21 of 26 datasets — and the constraint becomes attainability, not quality.

**Then choose by the asymmetry of costs.** Where a missed positive is expensive relative to a review — fraud, safety, AML — the coverage-first composer recovers the most positives per unit of capacity and, at 5%, is statistically indistinguishable from an unconstrained gradient-boosting model on recall. Where a false positive is expensive relative to a miss — customer-facing friction, account restriction — FIGS's 11-point mean precision advantage at 5% is the better trade, accepting that it will not spend the whole budget, that the advantage does not hold on every dataset, and that it is not statistically significant under the correction we apply to recall.

**Report the operating point, the pool and the objective.** Our own single-seed conclusions did not survive replication (§6.4), and our unmatched comparisons inverted under matching (§6.2). Both failures are avoidable by reporting realised alert rate, candidate-pool size, and composition objective alongside quality metrics.

**Deployment considerations.** An alert budget allocates scarce review capacity, and a rule set that concentrates alerts on a subpopulation concentrates review, investigation and adverse action on it. Rule-level auditability helps here — each condition is inspectable, and coverage can be measured per group — but it does not by itself establish equitable allocation. We report no fairness analysis, and any deployment in a regulated decision workflow should measure per-group alert rates and outcomes, not only aggregate precision and recall. This is an omission of the present study rather than a property of the method.

---

## 8. Limitations

**Statistical power remains limited at tight budgets.** The 5% comparison is significant at $n = 17$, but the 2% comparison ($n = 8$, $p = 0.077$) and the 1% comparison are not, because the common feasible subset shrinks as the budget tightens. The tight-budget regime is precisely where alert constraints bind hardest in practice, so the claims there rest on the descriptive feasibility counts of Table 3 rather than on a significance test. Note also that the number of *datasets* is the unit of analysis: our five seeds stabilise each cell but do not increase the power of the Friedman test.

**Dataset composition, and the robustness of the headline result.** Seven of twenty-seven datasets are NASA MDP software-defect data with documented quality issues [27]. Excluding them costs the 5% result its significance ($p = 0.154$ at $n = 11$). Four of the five paired mean differences are preserved or larger on that subset and the fifth is essentially unchanged, though two of the five effect sizes fall (Table 4b), and bootstrap power there is only 0.53. We therefore read it as predominantly a power loss rather than a contradiction, but a stricter reader may note that the *significant* result requires retaining that family, and we cannot rule this out with the datasets available. The fix is roughly six to nine further non-defect imbalanced datasets, which would take the ex-NASA sample to $n \approx 17$–20 and power to 0.89–0.96; it is not a different test. More importantly, the benchmark remains narrow relative to the motivating domain: `creditcard` is the only fraud dataset, and several of the added datasets (`diabetes`, `ionosphere`, `spambase`, `credit_g`) are near-balanced, where an alert budget is not the binding operational constraint. A benchmark motivated by fraud and abuse triage should be dominated by fraud and abuse data at realistic base rates, and ours is not.

**Baseline tuning.** Baselines run at library defaults against a 24-configuration generation grid (§5.2). The comparison is not budget-matched in tuning effort, which favours the proposed method on pool size (§6.6).

**Ordered models.** RIPPER and BRL are compared under a flattened disjunctive reading that is not native to them, and their availability under our quality conditions is low (§6.8). Truncating an ordered list to the alert budget is the correct comparison and is not implemented.

**Missing comparators.** BRCG [16] and Bayesian Rule Sets [17] learn the same DNF model class under complexity constraints and are the most informative missing baselines. CORELS could not be constructed in our environment (§6.8).

**The exact composer is under-evaluated.** One dataset, one number, reported on the selection split, and at an unmatched realised alert rate (§6.7). No scalability curve, no timeout characterisation, no distribution of the greedy gap. The formulation and bound of §3 are stated for the record; their empirical payoff is not established here.

**The pool-size experiment is unmatched.** §6.6 varies candidate-pool size at a fixed *requested* target, and realised alert rate falls with the cap, so the recall effect is confounded with the operating-point change. The experiment needs repeating at matched realised alert rate before the size of the effect can be stated.

**The generation-grid negative result is itself underpowered.** §6.4 uses 11 of the 27 datasets and 3 seeds. Failure to detect an effect at that sample size is not evidence of no effect, and the honest statement is that any effect of generation-grid design is smaller than this design can resolve — while the 67% compute cost of the expanded grid is certain.

**The tree-versus-non-tree generation comparison (§6.9) uses a different, unmatched protocol.** It is not run under §5.3's matched-realised-alert-rate protocol, so it cannot be combined with Tables 2–4, and its non-tree family's availability is uneven (0–75% failure by model, §6.9), so each family mean is a selection over datasets rather than a fixed sample. BRL's parameters there were tuned for tractability (to avoid multi-minute-per-fold mining blowups on class-balanced datasets), not for accuracy, so its contribution likely understates what a properly tuned BRL could achieve. Repeating §6.9 under the matched-budget protocol, with a tuned BRL, is future work.

**Table 2's row for RuleFit is over 26 datasets**, not 27, because it produced nothing on `creditcard`; the mean-reach column is therefore not strictly comparable across rows.

**Matching is on capacity, not on every realised operating point.** Where a method saturates, Table 4 credits it with its ceiling (§5.3). This is the right operational reading, but it means the headline recall comparison is partly between the proposed composer at 5% and saturating methods at their maxima; the counts are given in the Table 4 caption.

**Generation mechanisms are not separated.** All arms in §6.4 vary both class weighting and sample weighting; the factorial that would isolate them is absent, so the claim that they are complementary diversity controls is unsupported and we do not make it.

**Selection is not disjoint from generation** in the reported runs (§5.1). Test estimates are unbiased, but the measured 0.038 F1 selection optimism means selection-split figures, including §6.7, should not be read as generalisation estimates.

---

## 9. Conclusion

For detection systems whose output is a work queue, deployability is governed by an alert budget, and we have argued that this changes both how rule sets should be composed and how they should be compared. Formulated as maximum positive coverage of a disjunction under a realised-alert constraint, the problem has submodular structure in both objective and constraint, which removes the guarantee that would justify greedy composition and motivates the exact branch-and-bound composer and admissible bound we give.

Empirically, the finding we consider most useful is also the simplest: rule learners differ as much in whether they can hit a given alert budget as in their quality once matched to it. FIGS cannot spend a 5% review budget on 12 of 27 datasets, and RuleFit cannot operate at 1% on 21 of 26. Once matched on available capacity, the proposed composer ranks first on recall, significantly ahead of RuleFit, SkopeRules and FIGS under Holm-corrected pairwise tests, and statistically indistinguishable from an unconstrained gradient-boosting reference — the only rule learner here to reach that ceiling — while conceding 11 precision points to FIGS. At tighter budgets the evidence thins: at 2% the top two by rank are preserved but nothing reaches significance, and at 1% the feasible subset is three datasets. Removing the NASA MDP family also costs the 5% result its significance, at a sample size where the test has power 0.53 and where the paired differences largely persist, so we present the finding as established on this benchmark rather than as a general property of the method.

Two negative results are worth as much as the positive one. Generation-grid design, which we expected to matter, has no detectable effect on deployed quality while the expanded grid costs 67% more fitted trees. And our own earlier, unmatched comparisons inverted once realised alert rates were matched. Rule-system evaluation that does not report the realised operating point, the candidate-pool size and the composition objective is not comparing what it appears to compare.

The formulation, the admissible bound, the greedy and exact composers, and the benchmark harness that produced every table above are available in Iguanas, an open-source Python library [15], so that both the method and the protocol can be applied to alert budgets and datasets other than ours. We would regard the protocol as the more portable of the two: any rule learner can be placed on a realised-alert-rate curve, and doing so is what changed our own conclusions.

---

## Declarations

**Funding.** To be completed at submission.

**Declaration of competing interest.** The authors are employed by PayPal, which maintains Iguanas as an open-source project. The benchmark harness, baseline adapters and evaluation protocol were implemented by the authors. This is disclosed because the study evaluates software the authors maintain; §5.2 and §6.6 state the specific respects in which the comparison favours the proposed method.

**CRediT authorship contribution statement.** To be completed at submission.

**Data availability.** All datasets are public and are retrieved from OpenML by name and version through the harness registry (Table 1), with raw frames cached locally for offline reproduction.

**Code availability.** The implementation, benchmark harness and analysis scripts are in the open-source project repository [15]. Result artifacts, the five-seed raw results, matched-alert-rate tables, saturation ceilings and operating curves are versioned with the manuscript.

---

## References

1. Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence. 2019;1:206–215. https://doi.org/10.1038/s42256-019-0048-x
2. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. KDD 2016;785–794. https://doi.org/10.1145/2939672.2939785
3. Friedman JH, Popescu BE. Predictive learning via rule ensembles. Annals of Applied Statistics. 2008;2(3):916–954. https://doi.org/10.1214/07-AOAS148
4. Letham B, Rudin C, McCormick TH, Madigan D. Interpretable classifiers using rules and Bayesian analysis. Annals of Applied Statistics. 2015;9(3):1350–1373. https://doi.org/10.1214/15-AOAS848
5. Angelino E, Larus-Stone N, Alabi D, Seltzer M, Rudin C. Learning certifiably optimal rule lists. KDD 2017;35–44. https://doi.org/10.1145/3097983.3098047
6. Lin J, Zhong C, Hu D, Rudin C, Seltzer M. Generalized and scalable optimal sparse decision trees. ICML 2020;PMLR 119:6150–6160.
7. Tan YS, Singh C, Nasseri K, Agarwal A, Duncan J, Ronen O, et al. Fast interpretable greedy-tree sums. PNAS. 2025;122(7):e2310151122. https://doi.org/10.1073/pnas.2310151122
8. Cohen WW. Fast effective rule induction. ICML 1995;115–123.
9. Lakkaraju H, Bach SH, Leskovec J. Interpretable decision sets. KDD 2016;1675–1684. https://doi.org/10.1145/2939672.2939874
10. Wang F, Rudin C. Falling rule lists. AISTATS 2015;PMLR 38:1013–1022.
11. Deng H. Interpreting tree ensembles with inTrees. International Journal of Data Science and Analytics. 2019;7:277–287. https://doi.org/10.1007/s41060-018-0144-8
12. Bénard C, Biau G, da Veiga S, Scornet E. SIRUS: Stable and interpretable rule set for classification. Electronic Journal of Statistics. 2021;15:427–505. https://doi.org/10.1214/20-EJS1792
13. Liu B, Mazumder R. FIRE: An optimization approach for fast interpretable rule extraction. KDD 2023;1396–1405. https://doi.org/10.1145/3580305.3599353
14. Rudin C, Chen C, Chen Z, Huang H, Semenova L, Zhong C. Interpretable machine learning: Fundamental principles and 10 grand challenges. Statistical Surveys. 2022;16:1–85. https://doi.org/10.1214/21-SS133
15. PayPal. Iguanas: A Python library for rule generation and evaluation. https://github.com/paypal/Iguanas/
16. Dash S, Günlük O, Wei D. Boolean decision rules via column generation. NeurIPS 2018;4655–4665.
17. Wang T, Rudin C, Doshi-Velez F, Liu Y, Klampfl E, MacNeille P. A Bayesian framework for learning rule sets for interpretable classification. Journal of Machine Learning Research. 2017;18(70):1–37.
18. Malioutov D, Varshney KR. Exact rule learning via Boolean compressed sensing. ICML 2013;PMLR 28:765–773.
19. Lavrač N, Kavšek B, Flach P, Todorovski L. Subgroup discovery with CN2-SD. Journal of Machine Learning Research. 2004;5:153–188.
20. van Leeuwen M, Knobbe A. Diverse subgroup set discovery. Data Mining and Knowledge Discovery. 2012;25:208–242. https://doi.org/10.1007/s10618-012-0273-y
21. Nemhauser GL, Wolsey LA, Fisher ML. An analysis of approximations for maximizing submodular set functions. Mathematical Programming. 1978;14:265–294. https://doi.org/10.1007/BF01588971
22. Feige U. A threshold of ln n for approximating set cover. Journal of the ACM. 1998;45(4):634–652. https://doi.org/10.1145/285055.285059
23. Iyer RK, Bilmes JA. Submodular optimization with submodular cover and submodular knapsack constraints. NeurIPS 2013;2436–2444.
24. Bahnsen AC, Aouada D, Ottersten B. Example-dependent cost-sensitive decision trees. Expert Systems with Applications. 2015;42(19):6609–6619. https://doi.org/10.1016/j.eswa.2015.04.042
25. Dal Pozzolo A, Boracchi G, Caelen O, Alippi C, Bontempi G. Credit card fraud detection: A realistic modeling and a novel learning strategy. IEEE Transactions on Neural Networks and Learning Systems. 2018;29(8):3784–3797. https://doi.org/10.1109/TNNLS.2017.2736643
26. Demšar J. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research. 2006;7:1–30.
27. Shepperd M, Song Q, Sun Z, Mair C. Data quality: Some comments on the NASA software defect datasets. IEEE Transactions on Software Engineering. 2013;39(9):1208–1215. https://doi.org/10.1109/TSE.2013.11
28. Cohen WW, Singer Y. A simple, fast, and effective rule learner. AAAI/IAAI 1999;335–342.
29. Holte RC. Very simple classification rules perform well on most commonly used datasets. Machine Learning. 1993;11(1):63–90. https://doi.org/10.1023/A:1022631118932

---

## Figures to produce before submission

1. **Pipeline schematic** — generation grid → candidate pool → filter/dedup → budgeted composition → deployed rule set.
2. **Operating curves** — recall vs realised alert rate per dataset, one panel per dataset, all models, with saturation ceilings marked. This is the paper's central argument and must be a figure.
3. **Budget-attainment summary** — Table 3 as a stacked bar chart (hits / overshoot / saturated) per model per budget.
4. **Critical-difference diagram** — Friedman/Nemenyi at matched 5%, showing the non-significance honestly.
5. **Recall gap to the black-box ceiling** — §6.3 as a grouped bar chart across budgets.
6. **Branch-and-bound scalability** — solve time and nodes expanded vs candidate-pool size, once §6.7 is completed.
