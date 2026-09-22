"""Extract interpretable rules from fitted XGBoost / LightGBM / RandomForest models.

The generation step itself is standard decision-path extraction: a tree-based
model (gradient boosted or bagged) is fitted, and a root-to-leaf path from
each tree is serialised into a rule string. This is a well-established
technique, not a new rule-induction algorithm, and no claim of optimality or
completeness is made over the space of possible rules — the rules returned
are exactly those the model happened to build.

Two aspects shape *which* path is taken and *how diverse* the resulting rule
set is:

- **Monotone-constraint-guided traversal.** When every feature carries a
  monotone constraint of +1 or -1, :func:`extract_rule_with_monotone_constraints`
  walks root-to-leaf choosing the branch implied by each feature's constraint
  direction, rather than following the highest-gain leaf. This yields rules
  whose conditions all point in the business-expected direction. Otherwise,
  ``leaf_selection`` picks between :func:`extract_max_gain_rule` (one rule per
  tree, the single highest-gain leaf) and :func:`extract_positive_gain_rules`
  (zero or more rules per tree, one for every leaf that favours the positive
  class).
- **Sample-weight and ``scale_pos_weight`` steering.** The grid search refits
  the model across a grid of sample-weight schedules and ``scale_pos_weight``
  values. Each combination reshapes the loss surface (or, for
  RandomForestClassifier, which has no ``scale_pos_weight`` parameter, the
  equivalent ``class_weight={0: 1.0, 1: scale_pos_weight}``), so different
  splits win and different rules are extracted; the grid is a diversity
  mechanism, not a hyperparameter optimiser — results are pooled and
  deduplicated, not ranked.

Execution model
---------------
The grid search runs on a single node. Parallelism is provided by
:class:`joblib.Parallel` with the ``"threading"`` backend, so speed-up comes
from the booster releasing the GIL during fitting. There is no multiprocessing,
no cluster/distributed execution (no Dask, Ray or Spark), and no GPU-specific
code path; scale is bounded by one machine's cores and memory.
"""
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .rule_formatting import simplify_rule

# ---------------------------------------------------------------------------
# Booster-agnostic helpers
# ---------------------------------------------------------------------------

def _detect_booster_type(estimator: Any) -> str:
    """Return ``"lightgbm"``, ``"randomforest"`` or ``"xgboost"`` for the estimator."""
    if isinstance(estimator, RandomForestClassifier):
        return "randomforest"
    return "lightgbm" if type(estimator).__module__.startswith("lightgbm") else "xgboost"


def _normalise_lgbm_tree_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map a single-tree LightGBM ``trees_to_dataframe()`` slice to the XGBoost schema.

    After normalisation the tree can be passed unchanged to
    :func:`extract_max_gain_rule` and
    :func:`extract_rule_with_monotone_constraints`.

    Column mapping
    --------------
    tree_index  → Tree
    node_index  → ID   (string e.g. "0-S0", "0-L1")
    position    → Node (0-based int, reset per-tree)
    split_feature / "Leaf" → Feature
    threshold   → Split (float; NaN for leaves)
    left_child  → Yes  (condition-met child)
    right_child → No   (condition-not-met child)
    split_gain / value → Gain
    count       → Cover

    Notes
    -----
    Leaf nodes are identified by ``split_feature.isna()`` (LightGBM 4.x has no
    ``node_type`` column; leaves simply have no split feature).

    LightGBM uses ``<=`` for numeric splits; XGBoost uses ``<``.  Because tree
    thresholds are chosen to minimise error on continuous features, the
    distinction is irrelevant for practical rule strings, so the existing
    ``<``/``>=`` operators in the extraction functions are reused unchanged.
    """
    df = df.reset_index(drop=True)
    is_leaf = df["split_feature"].isna()  # leaves have no split feature
    return pd.DataFrame(
        {
            "Tree":    df["tree_index"],
            "Node":    df.index,                                          # sequential int per tree
            "ID":      df["node_index"],                                  # e.g. "0-S0", "0-L1"
            "Feature": df["split_feature"].where(~is_leaf, other="Leaf"),
            "Split":   pd.to_numeric(df["threshold"], errors="coerce"),
            "Yes":     df["left_child"],                                  # condition-met child
            "No":      df["right_child"],                                 # condition-not-met child
            # Leaf nodes: use raw prediction value; split nodes: information gain
            "Gain":    df["split_gain"].where(~is_leaf, other=df["value"]),
            "Cover":   df["count"],
        }
    )


def _rf_tree_to_dataframe(tree_idx: int, tree: Any, feature_names: list[str]) -> pd.DataFrame:
    """Convert one scikit-learn ``DecisionTreeClassifier.tree_`` to the canonical schema.

    Mirrors :func:`_normalise_lgbm_tree_df` for RandomForestClassifier's
    per-tree Cython ``tree_`` structure (``feature``, ``threshold``,
    ``children_left``/``children_right``, ``value``, ``impurity``,
    ``n_node_samples``), so a single tree can be passed unchanged to
    :func:`extract_max_gain_rule`, :func:`extract_positive_gain_rules` and
    :func:`extract_rule_with_monotone_constraints`.

    Leaf nodes are identified by ``children_left == -1`` (scikit-learn's
    ``TREE_LEAF`` sentinel). A leaf's ``Gain`` is the fraction of
    positive-class samples reaching it, minus 0.5, so positive values mean
    the leaf favours the positive class -- matching the sign convention of
    XGBoost/LightGBM raw leaf scores. A split's ``Gain`` is the weighted
    impurity decrease; kept for schema parity, not consumed by any extractor.
    """
    feature_idx = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    n_node_samples = tree.n_node_samples
    impurity = tree.impurity
    value = tree.value  # shape (n_nodes, 1, n_classes), already class fractions

    rows = []
    for node_id in range(tree.node_count):
        if children_left[node_id] == -1:  # leaf
            class_fracs = value[node_id, 0]
            total = class_fracs.sum()
            pos_frac = float(class_fracs[-1] / total) if total > 0 else 0.0
            rows.append(
                {
                    "Tree": tree_idx,
                    "Node": node_id,
                    "ID": f"{tree_idx}-{node_id}",
                    "Feature": "Leaf",
                    "Split": np.nan,
                    "Yes": None,
                    "No": None,
                    "Gain": pos_frac - 0.5,
                    "Cover": int(n_node_samples[node_id]),
                }
            )
            continue

        left, right = int(children_left[node_id]), int(children_right[node_id])
        parent_n = n_node_samples[node_id]
        gain = impurity[node_id] - (
            (n_node_samples[left] / parent_n) * impurity[left]
            + (n_node_samples[right] / parent_n) * impurity[right]
        )
        rows.append(
            {
                "Tree": tree_idx,
                "Node": node_id,
                "ID": f"{tree_idx}-{node_id}",
                "Feature": feature_names[feature_idx[node_id]],
                "Split": float(threshold[node_id]),
                "Yes": f"{tree_idx}-{left}",
                "No": f"{tree_idx}-{right}",
                "Gain": float(gain),
                "Cover": int(parent_n),
            }
        )
    return pd.DataFrame(rows)


def _get_trees_dataframe(estimator: Any) -> pd.DataFrame:
    """Return the canonical per-node tree table for XGBoost, LightGBM or RandomForest.

    XGBoost exposes it via ``estimator._Booster.trees_to_dataframe()``;
    LightGBM exposes it via ``estimator.booster_.trees_to_dataframe()``
    (raw schema, normalised later per-tree in :func:`extract_rules`);
    RandomForestClassifier has no such method, so each
    ``estimator.estimators_[i].tree_`` is converted directly to the
    canonical schema via :func:`_rf_tree_to_dataframe` and concatenated.
    """
    booster_type = _detect_booster_type(estimator)
    if booster_type == "lightgbm":
        return estimator.booster_.trees_to_dataframe()
    if booster_type == "randomforest":
        feat_names_in = getattr(estimator, "feature_names_in_", None)
        feature_names = (
            list(feat_names_in)
            if feat_names_in is not None
            else [f"f{i}" for i in range(estimator.n_features_in_)]
        )
        return pd.concat(
            [
                _rf_tree_to_dataframe(i, tree.tree_, feature_names)
                for i, tree in enumerate(estimator.estimators_)
            ],
            ignore_index=True,
        )
    return estimator._Booster.trees_to_dataframe()


def _get_monotone_constraints_dict(estimator: Any) -> dict[str, int]:
    """Return a ``{feature: constraint}`` dict for XGBoost or LightGBM.

    XGBoost stores constraints as a ``dict``; LightGBM stores them as a
    ``list`` indexed by feature position.  This helper normalises both
    formats to a single ``{feature_name: ±1}`` mapping so the rest of the
    extraction code is booster-agnostic.

    Parameters
    ----------
    estimator : XGBClassifier | LGBMClassifier | RandomForestClassifier
        Fitted estimator with monotone constraints already verified to be
        non-zero for every feature.

    Returns
    -------
    dict[str, int]
        Mapping from feature name to constraint value (``+1`` or ``-1``).
    """
    booster_type = _detect_booster_type(estimator)
    if booster_type == "randomforest":
        feat_names = list(estimator.feature_names_in_)
        return {
            name: int(c)
            for name, c in zip(feat_names, estimator.monotonic_cst, strict=False)
            if int(c) != 0
        }
    constraints_raw = estimator.monotone_constraints
    if booster_type == "lightgbm":
        if isinstance(constraints_raw, list | tuple):
            feat_names = estimator.booster_.feature_name()
            return {
                name: int(c)
                for name, c in zip(feat_names, constraints_raw, strict=False)
                if int(c) != 0
            }
        return cast(dict[str, int], dict(constraints_raw))
    # XGBoost: already a dict
    return cast(dict[str, int], constraints_raw)


# ---------------------------------------------------------------------------


def extract_max_gain_rule(tree_X: pd.DataFrame) -> str:
    """Extract the rule path to the leaf with maximum gain using bottom-to-top approach.

    Finds the leaf node with highest gain value and traces back to the root node,
    building the rule by reconstructing conditions from child to parent.

    Parameters
    ----------
    tree_X : pd.DataFrame
        Output from estimator._Booster.trees_to_dataframe() filtered for a single tree.
        Required columns: Tree, Node, ID, Feature, Split, Yes, No, Missing, Gain, Cover.

    Returns
    -------
    str
        Rule string in format (X["feat1"] >= Split1) & (X["feat2"] < Split2).
        Returns empty string if tree is empty or has no valid leaves.
    """
    if tree_X.empty:
        return ""

    # Find leaves (nodes with Feature == 'Leaf')
    leaves = tree_X[tree_X["Feature"] == "Leaf"]
    if leaves.empty:
        return ""

    # Find best leaf by gain
    best_idx = leaves["Gain"].idxmax()
    best_leaf_node = int(leaves.loc[best_idx, "Node"])

    # Index by ID for O(1) lookups
    tree_X = tree_X.set_index("ID")

    # Get starting node
    node_rows = tree_X[tree_X["Node"] == best_leaf_node]
    if node_rows.empty:
        return ""

    current_id = node_rows.index[0]
    root_id = tree_X["Tree"].iloc[0]

    # Build lookup dictionaries using itertuples for faster iteration
    yes_lookup = {}
    no_lookup = {}
    for row in tree_X.itertuples(index=True):
        if pd.notna(row.Yes):
            yes_lookup[row.Yes] = {
                "id": row.Index,
                "feature": row.Feature,
                "split": row.Split,
            }
        if pd.notna(row.No):
            no_lookup[row.No] = {
                "id": row.Index,
                "feature": row.Feature,
                "split": row.Split,
            }

    # Trace path from node back to root (bottom-to-top)
    conditions = []
    # root_id is the Tree column's int value, current_id is always a string
    # "{tree}-{node}" ID, so this can never be False on entry; the loop's real
    # exit is the `else: break` below once the walk reaches the actual root.
    while current_id != root_id:  # pragma: no branch
        # Find the parent node (which node has current_id as Yes or No child)
        if current_id in yes_lookup:
            parent = yes_lookup[current_id]
            conditions.append(f'(X["{parent["feature"]}"] < {round(parent["split"], 5)})')
            current_id = parent["id"]
        elif current_id in no_lookup:
            parent = no_lookup[current_id]
            conditions.append(f'(X["{parent["feature"]}"] >= {round(parent["split"], 5)})')
            current_id = parent["id"]
        else:
            break

    conditions.reverse()
    return " & ".join(conditions) if conditions else ""


def extract_positive_gain_rules(tree_X: pd.DataFrame) -> list[str]:
    """Extract every root-to-leaf path whose leaf favours the positive class.

    Unlike :func:`extract_max_gain_rule`, which returns only the single
    highest-gain leaf per tree, this walks the tree top-to-bottom once and
    collects every leaf whose value is positive -- so a single tree can
    contribute zero, one, or several rules, mirroring how a boosted ensemble's
    positive-leaf paths are pooled when treated as a flat disjunctive rule set.

    Parameters
    ----------
    tree_X : pd.DataFrame
        Output from estimator._Booster.trees_to_dataframe() filtered for a
        single tree. Required columns: Node, ID, Feature, Split, Yes, No, Gain.

    Returns
    -------
    list[str]
        Rule strings in format (X["feat1"] >= Split1) & (X["feat2"] < Split2),
        one per qualifying leaf. Empty list if the tree has no positive leaf.
    """
    if tree_X.empty:
        return []

    tree_X = tree_X.set_index("ID")
    root_rows = tree_X[tree_X["Node"] == 0]
    if root_rows.empty:
        return []
    root_id = root_rows.index[0]

    rules: list[str] = []

    def walk(node_id: Any, path: list[str]) -> None:
        if node_id not in tree_X.index:
            return
        row = tree_X.loc[node_id]
        if row["Feature"] == "Leaf":
            if pd.notna(row["Gain"]) and float(row["Gain"]) > 0 and path:
                rules.append(" & ".join(path))
            return
        feature, split = row["Feature"], round(row["Split"], 5)
        walk(row["Yes"], [*path, f'(X["{feature}"] < {split})'])
        walk(row["No"], [*path, f'(X["{feature}"] >= {split})'])

    walk(root_id, [])
    return rules


def extract_rule_with_monotone_constraints(
    tree_X: pd.DataFrame, monotone_constraints: dict[str, int]
) -> str:
    """Extract rule path following monotone constraints using top-to-bottom approach.

    Starts from root and follows tree structure based on monotone constraints.
    NOTE: Only applicable if ALL features have a monotone constraint of -1 or +1.
    Features with constraint 0 will raise a ValueError.

    Parameters
    ----------
    tree_X : pd.DataFrame
        Output from estimator._Booster.trees_to_dataframe() filtered for a single tree.
        Required columns: Tree, Node, ID, Feature, Split, Yes, No, Missing.
    monotone_constraints : dict[str, int]
        Dictionary mapping feature names to constraint values:

        - +1 (positive): follow "No" branch (feature >= threshold)
        - -1 (negative): follow "Yes" branch (feature < threshold)
        - 0 (none): raises ValueError - not supported

    Returns
    -------
    str
        Rule string in format (X["feat1"] >= Split1) & (X["feat2"] < Split2).
        Returns empty string if tree is empty or starts with a leaf.

    Raises
    ------
    ValueError
        If a feature has no constraint defined or has constraint 0.
    """
    current_node = tree_X[tree_X["Node"] == 0]
    if current_node.empty:
        return ""

    conditions = []

    # Traverse from root to leaf (top-to-bottom)
    while True:
        current_node_data = current_node.iloc[0]
        feature = current_node_data["Feature"]

        # Stop if we've reached a leaf
        if feature == "Leaf":
            break

        split_value = round(current_node_data["Split"], 5)
        constraint = monotone_constraints.get(feature, 0)

        # Follow branch based on monotone constraint
        if constraint == 1:
            # Positive constraint: feature >= threshold (follow "No" branch)
            conditions.append(f'(X["{feature}"] >= {split_value})')
            next_id = current_node_data["No"]
        elif constraint == -1:
            # Negative constraint: feature < threshold (follow "Yes" branch)
            conditions.append(f'(X["{feature}"] < {split_value})')
            next_id = current_node_data["Yes"]
        else:
            raise ValueError(
                f"Feature '{feature}' has no monotone constraint defined or has constraint 0. "
                f"Please provide a constraint of +1 or -1 for all features in the tree."
            )

        # Move to next node
        current_node = tree_X[tree_X["ID"] == next_id]
        if current_node.empty:
            break

    return " & ".join(conditions) if conditions else ""


def extract_rules(
    estimator: XGBClassifier,
    all_features_constrained: bool,
    leaf_selection: str = "max_gain",
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate rules extracted from XGBoost, LightGBM or RandomForest trees.

    Parameters
    ----------
    estimator : XGBClassifier | LGBMClassifier | RandomForestClassifier
        Fitted tree-based classifier. XGBoost, LightGBM and scikit-learn's
        RandomForestClassifier are supported.
    all_features_constrained : bool
        If True, uses monotone constraint-based extraction (top-to-bottom),
        which always yields one rule per tree; ``leaf_selection`` is ignored.
        If False, ``leaf_selection`` picks the extraction strategy.
    leaf_selection : {"max_gain", "all_positive"}, default="max_gain"
        Strategy used when ``all_features_constrained`` is False:

        - ``"max_gain"``: one rule per tree, the single highest-gain leaf
          (see :func:`extract_max_gain_rule`).
        - ``"all_positive"``: zero or more rules per tree, one for every leaf
          whose value favours the positive class (see
          :func:`extract_positive_gain_rules`).
    **kwargs : dict
        Additional metadata columns added to the output DataFrame
        (e.g., transformation name, scale_pos_weight value).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``rule``, ``tree``, and any ``kwargs`` columns.

    Raises
    ------
    ValueError
        If ``leaf_selection`` is not ``"max_gain"`` or ``"all_positive"``.
    """
    if leaf_selection not in ("max_gain", "all_positive"):
        raise ValueError(
            f"leaf_selection must be 'max_gain' or 'all_positive', got {leaf_selection!r}"
        )

    booster_type = _detect_booster_type(estimator)
    df = _get_trees_dataframe(estimator)
    group_col = "tree_index" if booster_type == "lightgbm" else "Tree"

    rule_strings = []
    tree_ids = []

    for tree_id, tree in df.groupby(group_col, sort=False):
        if tree.empty:
            continue

        # Normalise to the XGBoost canonical column schema
        if booster_type == "lightgbm":
            tree = _normalise_lgbm_tree_df(tree)
        else:
            tree = tree.reset_index(drop=True)

        if all_features_constrained:
            mc_dict = _get_monotone_constraints_dict(estimator)
            rule = extract_rule_with_monotone_constraints(tree, mc_dict)
            rules_for_tree = [simplify_rule(rule)] if rule else []
        elif leaf_selection == "all_positive":
            rules_for_tree = [
                simplified
                for rule in extract_positive_gain_rules(tree)
                if (simplified := simplify_rule(rule))
            ]
        else:
            rule = extract_max_gain_rule(tree)
            rules_for_tree = [simplify_rule(rule)] if rule else []

        for rule in rules_for_tree:
            rule_strings.append(rule)
            tree_ids.append(tree_id)

    if rule_strings:
        rules_data: dict[str, Any] = {"rule": rule_strings, "tree": tree_ids}
        for key, value in kwargs.items():
            rules_data[key] = [value] * len(rule_strings)
        return pd.DataFrame(rules_data)
    return pd.DataFrame()


def _check_all_features_have_monotone_constraints(
    estimator: XGBClassifier, n_features: int
) -> bool:
    """Check if all features have non-zero monotone constraints.

    Handles XGBoost (``dict``), LightGBM (``list``) and RandomForestClassifier
    (``monotonic_cst`` array) constraint formats.

    Parameters
    ----------
    estimator : XGBClassifier | LGBMClassifier | RandomForestClassifier
        The fitted estimator to inspect.
    n_features : int
        Expected number of features.

    Returns
    -------
    bool
        True if all n_features features have constraints of +1 or -1.
    """
    booster_type = _detect_booster_type(estimator)
    if booster_type == "randomforest":
        constraints = getattr(estimator, "monotonic_cst", None)
        if constraints is None:
            return False
        return len(constraints) == n_features and all(int(c) != 0 for c in constraints)
    if not getattr(estimator, "monotone_constraints", None):
        return False
    if booster_type == "lightgbm":
        constraints = estimator.monotone_constraints
        if isinstance(constraints, list | tuple):
            return len(constraints) == n_features and all(int(c) != 0 for c in constraints)
        if isinstance(constraints, dict):
            return len(constraints) == n_features and all(c != 0 for c in constraints.values())
        return False  # unknown constraint format
    # XGBoost: monotone_constraints is a dict
    if not isinstance(estimator.monotone_constraints, dict):
        return False
    return len(estimator.monotone_constraints) == n_features and all(
        constraint != 0 for constraint in estimator.monotone_constraints.values()
    )


def _apply_scale_pos_weight(est: Any, scale_pos_weight: float, estimator_params: dict[str, Any]) -> None:
    """Steer class balance for one grid-search fit.

    XGBoost/LightGBM expose ``scale_pos_weight`` directly. RandomForestClassifier
    has no such parameter, so the same steering effect is approximated via
    ``class_weight={0: 1.0, 1: scale_pos_weight}`` -- a larger weight makes the
    positive class relatively more influential on splits, mirroring what
    ``scale_pos_weight`` does for the boosted estimators.
    """
    if estimator_params.get("objective") == "binary:hinge":
        return
    if isinstance(est, RandomForestClassifier):
        est.set_params(class_weight={0: 1.0, 1: float(scale_pos_weight)})
    else:
        est.set_params(scale_pos_weight=scale_pos_weight)


def _train_rules_for_weight_transformation(
    weights: pd.Series | np.ndarray,
    estimator_params: dict[str, Any],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    scale_pos_weights: np.ndarray,
    all_features_constrained: bool,
    feature_names: list[str] | None = None,
    estimator_class: type = XGBClassifier,
    leaf_selection: str = "max_gain",
) -> list[pd.DataFrame]:
    """
    Process a single weight column across all scale_pos_weight values.

    This helper function is used for parallel execution in rule_grid_search.

    Parameters
    ----------
    weights : pd.Series | np.ndarray
        Sample weights for this transformation
    estimator_params : dict
        XGBoost estimator parameters to reconstruct the model
    X_train : pd.DataFrame | np.ndarray
        Training features as numpy array (serializes faster than DataFrame for IPC).
    y_train : pd.Series | np.ndarray
        Training target as numpy array.
    scale_pos_weights : np.ndarray
        Array of scale_pos_weight values to try
    all_features_constrained : bool
        Whether to use monotone constraint-based extraction
    feature_names : list[str] | None, default=None
        Original column names for X_train. When provided and X_train is a numpy
        array, a DataFrame is reconstructed inside the worker so that XGBoost
        preserves feature names (required for monotone-constraint rule extraction).
    leaf_selection : {"max_gain", "all_positive"}, default="max_gain"
        Forwarded to :func:`extract_rules`; ignored when all_features_constrained.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames with extracted rules
    """
    rules_dfs = []
    transformation = weights.name if hasattr(weights, "name") else "Baseline"
    weights_array = weights.values if hasattr(weights, "values") else weights

    # Reconstruct DataFrame from numpy + names so XGBoost preserves feature names
    # in the booster (needed for monotone-constraint extraction and readable rules).
    # This is cheap — the array is already deserialized; only metadata is created.
    if feature_names is not None and isinstance(X_train, np.ndarray):
        X_fit: pd.DataFrame | np.ndarray = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_fit = X_train

    for scale_pos_weight in scale_pos_weights:
        est = estimator_class(**estimator_params)
        _apply_scale_pos_weight(est, scale_pos_weight, estimator_params)
        try:
            _ = est.fit(X_fit, y_train, sample_weight=weights_array)
        except Exception:
            continue

        params = {
            "transformation": transformation,
            "scale_pos_weight": scale_pos_weight,
        }
        rules_df = extract_rules(est, all_features_constrained, leaf_selection, **params)

        if not rules_df.empty:
            rules_dfs.append(rules_df)

    return rules_dfs


def _train_rules_for_scale(
    scale_pos_weight: float,
    weights_np: np.ndarray,
    weight_columns: list[str],
    estimator_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    all_features_constrained: bool,
    feature_names: list[str] | None = None,
    estimator_class: type = XGBClassifier,
    leaf_selection: str = "max_gain",
) -> list[pd.DataFrame]:
    """
    Process all weight transformations for a single scale_pos_weight value.

    This helper function is used for parallel execution in rule_grid_search_parallel_scales.

    Parameters
    ----------
    scale_pos_weight : float
        The scale_pos_weight value to use for this run.
    weights_np : np.ndarray
        2D array of shape (n_samples, n_transformations) containing all weight columns.
    weight_columns : list[str]
        Names of the weight transformations (column labels for weights_np).
    estimator_params : dict
        XGBoost estimator parameters to reconstruct the model.
    X_train : np.ndarray
        Training features as numpy array.
    y_train : np.ndarray
        Training target as numpy array.
    all_features_constrained : bool
        Whether to use monotone constraint-based extraction.
    feature_names : list[str] | None, default=None
        Original column names for X_train. When provided, a DataFrame is
        reconstructed so that XGBoost preserves feature names.
    leaf_selection : {"max_gain", "all_positive"}, default="max_gain"
        Forwarded to :func:`extract_rules`; ignored when all_features_constrained.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames with extracted rules, one entry per weight
        transformation that produced at least one rule.
    """
    rules_dfs = []

    if feature_names is not None and isinstance(X_train, np.ndarray):
        X_fit: pd.DataFrame | np.ndarray = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_fit = X_train
    for i, name in enumerate(weight_columns):
        weights_array = weights_np[:, i]
        est = estimator_class(**estimator_params)
        _apply_scale_pos_weight(est, scale_pos_weight, estimator_params)
        try:
            est.fit(X_fit, y_train, sample_weight=weights_array)
        except Exception:
            continue

        params = {
            "transformation": name,
            "scale_pos_weight": scale_pos_weight,
        }
        rules_df = extract_rules(est, all_features_constrained, leaf_selection, **params)
        if not rules_df.empty:
            rules_dfs.append(rules_df)

    return rules_dfs


def _setup_and_validate_grid_search(
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weights: list[float] | np.ndarray,
    sample_weights_df: pl.DataFrame | pd.DataFrame | None = None,
    estimator: XGBClassifier | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, dict[str, Any], bool, type]:
    """Validate inputs and prepare data for grid search functions.

    Parameters
    ----------
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list[float] | np.ndarray
        Array of scale_pos_weight values to try.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
    estimator : XGBClassifier | None, default=None
        Estimator to check for monotone constraints.

    Returns
    -------
    tuple
        (X_train_np, y_train_np, feature_names, sample_weights_df_pd,
         estimator_params, all_features_constrained)
    """
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    feature_names = list(X_train.columns)

    if X_train_np.dtype == object:
        raise ValueError(
            "X_train contains non-numeric data. Please encode categorical features "
            "numerically before using rule_grid_search_parallel_scales."
        )

    if len(scale_pos_weights) == 0:
        raise ValueError("scale_pos_weights cannot be empty")

    if sample_weights_df is None:
        sample_weights_df_pd = pd.DataFrame({"Baseline": np.ones(len(X_train))})
    elif isinstance(sample_weights_df, pl.DataFrame):
        sample_weights_df_pd = sample_weights_df.to_pandas()
    else:
        sample_weights_df_pd = sample_weights_df

    estimator_class: type = XGBClassifier
    estimator_params = {}
    all_features_constrained = False
    if estimator is not None:
        estimator_class = type(estimator)
        estimator_params = estimator.get_params()
        estimator_params.pop("scale_pos_weight", None)
        n_features = len(X_train.columns)
        all_features_constrained = _check_all_features_have_monotone_constraints(
            estimator, n_features
        )

    return (
        X_train_np,
        y_train_np,
        feature_names,
        sample_weights_df_pd,
        estimator_params,
        all_features_constrained,
        estimator_class,
    )


def _finalize_grid_search_results(
    rules_dfs: list[pd.DataFrame],
    verbose: int = 0,
    context: str = "grid search",
) -> pl.DataFrame:
    """Concatenate, deduplicate, and convert rules to Polars DataFrame.

    Parameters
    ----------
    rules_dfs : list[pd.DataFrame]
        List of rule DataFrames to consolidate.
    verbose : int, default=0
        Verbosity level for output messages.
    context : str, default="grid search"
        Description of the search context for logging.

    Returns
    -------
    pl.DataFrame
        Deduplicated rules as a Polars DataFrame.
    """
    if rules_dfs:
        final_X_pd = pd.concat(rules_dfs, ignore_index=True)
        final_X = pl.from_pandas(final_X_pd)
    else:
        final_X = pl.DataFrame()

    final_X = final_X.unique("rule") if final_X.height > 0 else final_X
    if verbose > 0:
        print(f"Extracted {len(final_X)} total rules from {context}")

    return final_X


def rule_grid_search_sequential(
    estimator: XGBClassifier,
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weights: list[float] | np.ndarray,
    sample_weights_df: pl.DataFrame | pd.DataFrame | None = None,
    verbose: int = 0,
    leaf_selection: str = "max_gain",
) -> pl.DataFrame:
    """
    Sequential (single-threaded) variant of rule_grid_search.

    Identical behaviour to :func:`rule_grid_search` but runs the grid in a
    plain loop without joblib. Useful for debugging, for deterministic
    profiling, or for small workloads where the thread-dispatch overhead
    outweighs the benefit of parallelism.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list | np.ndarray
        Array of scale_pos_weight values to try.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
        If None, uses baseline weights of 1.0 for all samples.
    verbose : int, default=0
        Controls verbosity. 0 = silent, 1 = summary.
    leaf_selection : {"max_gain", "all_positive"}, default="max_gain"
        Forwarded to :func:`extract_rules` for every unconstrained tree; see
        that function for what each option does. Ignored for trees where every
        feature carries a monotone constraint.

    Returns
    -------
    pl.DataFrame
        Same schema as :func:`rule_grid_search`: columns rule, tree,
        scale_pos_weight, transformation.
    """
    (
        X_train_np,
        y_train_np,
        feature_names,
        sample_weights_df_pd,
        estimator_params,
        all_features_constrained,
        estimator_class,
    ) = _setup_and_validate_grid_search(
        X_train, y_train, scale_pos_weights, sample_weights_df, estimator
    )

    weight_columns = list(sample_weights_df_pd.columns)
    weights_np = sample_weights_df_pd.to_numpy()

    if verbose > 0:
        print(
            f"Starting sequential rule grid search with {len(weight_columns)} weight "
            f"transformations and {len(scale_pos_weights)} scale_pos_weight values "
            f"({len(weight_columns) * len(scale_pos_weights)} total combinations)"
        )

    rules_dfs = []
    for scale_pos_weight in scale_pos_weights:
        results = _train_rules_for_scale(
            scale_pos_weight,
            weights_np,
            weight_columns,
            estimator_params,
            X_train_np,
            y_train_np,
            all_features_constrained,
            feature_names=feature_names,
            estimator_class=estimator_class,
            leaf_selection=leaf_selection,
        )
        rules_dfs.extend(results)

    return _finalize_grid_search_results(rules_dfs, verbose, "sequential grid search")


def rule_grid_search_parallel_weights(
    estimator: XGBClassifier,
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weights: list[float] | np.ndarray,
    sample_weights_df: pl.DataFrame | pd.DataFrame | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
    leaf_selection: str = "max_gain",
) -> pl.DataFrame:
    """
    Grid search over sample weight transformations and scale_pos_weight values, parallelised over weight transformations.

    This function systematically trains XGBoost models with different combinations of:
    - sample weights
    - scale_pos_weight values

    For each combination, it extracts rules from the fitted models and returns them as a Polars DataFrame.
    The results from all combinations are pooled and deduplicated; the grid is a
    mechanism for producing a diverse candidate set, not a search for a single
    "best" configuration, and no optimality over the space of rules is claimed.
    The weight-transformation loop is parallelised with :class:`joblib.Parallel`
    using the ``"threading"`` backend — single-node, thread-parallel only.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list | np.ndarray
        Array of scale_pos_weight values to try. Iterated sequentially within
        each worker thread.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
        If None, uses baseline weights of 1.0 for all samples.
    n_jobs : int, default=-1
        Number of joblib worker threads. -1 means one per available core.
    verbose : int, default=0
        Controls the verbosity level:

        - 0: silent (no output)
        - 1: progress information (start/end summary)
        - >=2: detailed progress with live updates from joblib Parallel backend

    Returns
    -------
    pl.DataFrame
        Same schema as :func:`rule_grid_search`: columns rule, tree,
        scale_pos_weight, transformation.

    Examples
    --------
    >>> weights_train = generate_sample_weight_transformations(X_train["amount"])
    >>> scale_pos_weights = np.logspace(0, np.log10(imbalance_ratio*2), 20)
    >>> results = rule_grid_search(
    ...     estimator, X_train, y_train,
    ...     scale_weights, weights_train, n_jobs=-1, verbose=1
    ... )
    """
    (
        X_train_np,
        y_train_np,
        feature_names,
        sample_weights_df_pd,
        estimator_params,
        all_features_constrained,
        estimator_class,
    ) = _setup_and_validate_grid_search(
        X_train, y_train, scale_pos_weights, sample_weights_df, estimator
    )

    weight_columns = sample_weights_df_pd.columns
    joblib_verbose = 10 if verbose >= 2 else 0

    if verbose > 0:
        print(
            f"Starting rule grid search with {len(weight_columns)} weight transformations "
            f"and {len(scale_pos_weights)} scale_pos_weight values "
            f"({len(weight_columns) * len(scale_pos_weights)} total combinations)"
        )

    results_nested = Parallel(n_jobs=n_jobs, backend="threading", verbose=joblib_verbose)(
        delayed(_train_rules_for_weight_transformation)(
            sample_weights_df_pd[name],
            estimator_params,
            X_train_np,
            y_train_np,
            scale_pos_weights,
            all_features_constrained,
            feature_names,
            estimator_class,
            leaf_selection,
        )
        for name in weight_columns
    )

    rules_dfs = [rule_df for sublist in results_nested if sublist for rule_df in sublist]
    return _finalize_grid_search_results(rules_dfs, verbose, "grid search")


def rule_grid_search_parallel_scales(
    estimator: XGBClassifier,
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weights: list[float] | np.ndarray,
    sample_weights_df: pl.DataFrame | pd.DataFrame | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
    leaf_selection: str = "max_gain",
) -> pl.DataFrame:
    """
    Grid search parallelised over scale_pos_weight values.

    This function systematically trains XGBoost models with different combinations of:
    - sample weights
    - scale_pos_weight values

    For each combination, it extracts rules from the fitted models and returns them as a Polars DataFrame.
    The results from all combinations are pooled and deduplicated; the grid is a
    mechanism for producing a diverse candidate set, not a search for a single
    "best" configuration, and no optimality over the space of rules is claimed.
    The scale_pos_weight loop is parallelised with :class:`joblib.Parallel`
    using the ``"threading"`` backend — single-node, thread-parallel only.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list | np.ndarray
        Array of scale_pos_weight values to try. Distributed across worker threads.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
        If None, uses baseline weights of 1.0 for all samples.
    n_jobs : int, default=-1
        Number of joblib worker threads. -1 means one per available core.
    verbose : int, default=0
        Controls the verbosity level:

        - 0: silent (no output)
        - 1: progress information (start/end summary)
        - >=2: detailed progress with live updates from joblib Parallel backend

    Returns
    -------
    pl.DataFrame
        Same schema as :func:`rule_grid_search`: columns rule, tree,
        scale_pos_weight, transformation.
    """
    (
        X_train_np,
        y_train_np,
        feature_names,
        sample_weights_df_pd,
        estimator_params,
        all_features_constrained,
        estimator_class,
    ) = _setup_and_validate_grid_search(
        X_train, y_train, scale_pos_weights, sample_weights_df, estimator
    )

    weight_columns = list(sample_weights_df_pd.columns)
    weights_np = sample_weights_df_pd.to_numpy()
    joblib_verbose = 10 if verbose >= 2 else 0

    if verbose > 0:
        print(
            f"Starting parallel-scales rule grid search with {len(weight_columns)} weight "
            f"transformations and {len(scale_pos_weights)} scale_pos_weight values "
            f"({len(weight_columns) * len(scale_pos_weights)} total combinations)"
        )

    results_nested = Parallel(n_jobs=n_jobs, backend="threading", verbose=joblib_verbose)(
        delayed(_train_rules_for_scale)(
            scale_pos_weight,
            weights_np,
            weight_columns,
            estimator_params,
            X_train_np,
            y_train_np,
            all_features_constrained,
            feature_names,
            estimator_class,
            leaf_selection,
        )
        for scale_pos_weight in scale_pos_weights
    )

    rules_dfs = [rule_df for sublist in results_nested if sublist for rule_df in sublist]
    return _finalize_grid_search_results(rules_dfs, verbose, "parallel-scales grid search")


def rule_grid_search(
    estimator: XGBClassifier,
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weights: list[float] | np.ndarray,
    sample_weights_df: pl.DataFrame | pd.DataFrame | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
    leaf_selection: str = "max_gain",
) -> pl.DataFrame:
    """
    Grid search over scale_pos_weight values and sample weight transformations.

    Dispatches to :func:`rule_grid_search_parallel_scales` or
    :func:`rule_grid_search_parallel_weights` depending on which axis of the
    grid is larger, so that the parallelised loop is the longer one.

    This function systematically trains XGBoost models with different combinations of:
    - sample weights
    - scale_pos_weight values

    For each combination, it extracts rules from the fitted models and returns them as a Polars DataFrame.
    The results from all combinations are pooled and deduplicated; the grid is a
    mechanism for producing a diverse candidate set, not a search for a single
    "best" configuration, and no optimality over the space of rules is claimed.
    Parallelism is :class:`joblib.Parallel` with the ``"threading"`` backend —
    single-node, thread-parallel only.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list | np.ndarray
        Array of scale_pos_weight values to try.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
        If None, uses baseline weights of 1.0 for all samples.
    n_jobs : int, default=-1
        Number of joblib worker threads. -1 means one per available core.
    verbose : int, default=0
        Controls the verbosity level:

        - 0: silent (no output)
        - 1: progress information (start/end summary)
        - >=2: detailed progress with live updates from joblib Parallel backend
    leaf_selection : {"max_gain", "all_positive"}, default="max_gain"
        Forwarded to :func:`extract_rules` for every unconstrained tree.

    Returns
    -------
    pl.DataFrame
        Same schema as :func:`rule_grid_search`: columns rule, tree,
        scale_pos_weight, transformation.
    """
    if (
        len(scale_pos_weights) > len(sample_weights_df.columns)
        if sample_weights_df is not None
        else 1
    ):
        return rule_grid_search_parallel_scales(
            estimator, X_train, y_train, scale_pos_weights, sample_weights_df, n_jobs, verbose,
            leaf_selection,
        )
    else:
        return rule_grid_search_parallel_weights(
            estimator, X_train, y_train, scale_pos_weights, sample_weights_df, n_jobs, verbose,
            leaf_selection,
        )
