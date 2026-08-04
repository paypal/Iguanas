from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from xgboost import XGBClassifier

from .rule_formatting import simplify_rule


# ---------------------------------------------------------------------------
# Booster-agnostic helpers
# ---------------------------------------------------------------------------

def _detect_booster_type(estimator: Any) -> str:
    """Return ``"lightgbm"`` or ``"xgboost"`` based on the estimator's module."""
    return "lightgbm" if type(estimator).__module__.startswith("lightgbm") else "xgboost"


def _normalise_lgbm_tree_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map a single-tree LightGBM ``trees_to_dataframe()`` slice to the XGBoost schema.

    After normalisation the tree can be passed unchanged to
    :func:`extract_rule_by_max_gain` and
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


def _get_trees_dataframe(estimator: Any) -> pd.DataFrame:
    """Call the correct ``trees_to_dataframe()`` for XGBoost or LightGBM.

    XGBoost exposes it via ``estimator._Booster``;
    LightGBM exposes it via ``estimator.booster_``.
    """
    if _detect_booster_type(estimator) == "lightgbm":
        return estimator.booster_.trees_to_dataframe()
    return estimator._Booster.trees_to_dataframe()


def _get_monotone_constraints_dict(estimator: Any) -> dict[str, int]:
    """Return a ``{feature: constraint}`` dict for XGBoost or LightGBM.

    XGBoost stores constraints as a ``dict``; LightGBM stores them as a
    ``list`` indexed by feature position.  This helper normalises both
    formats to a single ``{feature_name: ±1}`` mapping so the rest of the
    extraction code is booster-agnostic.

    Parameters
    ----------
    estimator : XGBClassifier | LGBMClassifier
        Fitted estimator with monotone constraints already verified to be
        non-zero for every feature.

    Returns
    -------
    dict[str, int]
        Mapping from feature name to constraint value (``+1`` or ``-1``).
    """
    booster_type = _detect_booster_type(estimator)
    constraints_raw = estimator.monotone_constraints
    if booster_type == "lightgbm":
        if isinstance(constraints_raw, (list, tuple)):
            feat_names = estimator.booster_.feature_name()
            return {
                name: int(c)
                for name, c in zip(feat_names, constraints_raw)
                if int(c) != 0
            }
        return dict(constraints_raw)  # type: ignore[arg-type]
    # XGBoost: already a dict
    return constraints_raw  # type: ignore[return-value]


# ---------------------------------------------------------------------------


def extract_rule_by_max_gain(tree_X: pd.DataFrame) -> str:
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
    best_leaf_node = int(leaves.loc[best_idx, "Node"])  # type: ignore

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
    while current_id != root_id:
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
    **kwargs,
) -> pd.DataFrame:
    """Generate rules extracted from XGBoost or LightGBM trees.

    Parameters
    ----------
    estimator : XGBClassifier | LGBMClassifier
        Fitted tree-based classifier. Both XGBoost and LightGBM are supported.
    all_features_constrained : bool
        If True, uses monotone constraint-based extraction (top-to-bottom).
        If False, uses max gain-based extraction (bottom-to-top).
    **kwargs : dict
        Additional metadata columns added to the output DataFrame
        (e.g., transformation name, scale_pos_weight value).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``rule``, ``tree``, and any ``kwargs`` columns.
    """
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
            rule = simplify_rule(rule)
        else:
            rule = extract_rule_by_max_gain(tree)
            rule = simplify_rule(rule)

        if not rule:
            continue

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

    Handles both XGBoost (``dict``) and LightGBM (``list``) constraint formats.

    Parameters
    ----------
    estimator : XGBClassifier | LGBMClassifier
        The fitted estimator to inspect.
    n_features : int
        Expected number of features.

    Returns
    -------
    bool
        True if all n_features features have constraints of +1 or -1.
    """
    if not getattr(estimator, "monotone_constraints", None):
        return False
    booster_type = _detect_booster_type(estimator)
    if booster_type == "lightgbm":
        constraints = estimator.monotone_constraints
        if isinstance(constraints, (list, tuple)):
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


def _train_rules_for_weight_transformation(
    weights: pd.Series | np.ndarray,
    estimator_params: dict[str, Any],
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    scale_pos_weights: np.ndarray,
    all_features_constrained: bool,
    feature_names: list[str] | None = None,
    estimator_class: type = XGBClassifier,
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

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames with extracted rules
    """
    rules_dfs = []
    transformation = weights.name if hasattr(weights, "name") else "Baseline"  # type: ignore
    weights_array = weights.values if hasattr(weights, "values") else weights  # type: ignore

    # Reconstruct DataFrame from numpy + names so XGBoost preserves feature names
    # in the booster (needed for monotone-constraint extraction and readable rules).
    # This is cheap — the array is already deserialized; only metadata is created.
    if feature_names is not None and isinstance(X_train, np.ndarray):
        X_fit: pd.DataFrame | np.ndarray = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_fit = X_train

    for scale_pos_weight in scale_pos_weights:
        est = estimator_class(**estimator_params)
        if estimator_params.get("objective") != "binary:hinge":
            est.scale_pos_weight = scale_pos_weight
        try:
            _ = est.fit(X_fit, y_train, sample_weight=weights_array)
        except Exception:
            continue

        params = {
            "transformation": transformation,
            "scale_pos_weight": scale_pos_weight,
        }
        rules_df = extract_rules(est, all_features_constrained, **params)

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
        if estimator_params.get("objective") != "binary:hinge":
            est.set_params(scale_pos_weight=scale_pos_weight)
        est.fit(X_fit, y_train, sample_weight=weights_array)

        params = {
            "transformation": name,
            "scale_pos_weight": scale_pos_weight,
        }
        rules_df = extract_rules(est, all_features_constrained, **params)
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
) -> pl.DataFrame:
    """
    Sequential (single-process) variant of rule_grid_search.

    Identical behaviour to :func:`rule_grid_search` but runs in a single process
    without joblib parallelism. Useful for debugging, environments where
    multiprocessing is unavailable, or small workloads where process-spawn
    overhead outweighs the benefit of parallelism.

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
) -> pl.DataFrame:
    """
    Perform grid search over sample weight transformations and scale_pos_weight values to find optimal rules.

    This function systematically trains XGBoost models with different combinations of:
    - sample weights
    - scale_pos_weight values

    For each combination, it extracts rules from the fitted models and returns them as a Polars DataFrame.
    The weight transformations loop is parallelized using joblib for improved performance.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list | np.ndarray
        Array of scale_pos_weight values to try. Parallelised across workers.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
        If None, uses baseline weights of 1.0 for all samples.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
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
) -> pl.DataFrame:
    """
    Perform grid search parallelised over scale_pos_weight values.

    This function systematically trains XGBoost models with different combinations of:
    - sample weights
    - scale_pos_weight values

    For each combination, it extracts rules from the fitted models and returns them as a Polars DataFrame.
    The weight transformations loop is parallelized using joblib for improved performance.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list | np.ndarray
        Array of scale_pos_weight values to try. Parallelised across workers.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
        If None, uses baseline weights of 1.0 for all samples.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
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
) -> pl.DataFrame:
    """
    Perform grid search parallelised over scale_pos_weight values or sample_weights to find optimal rules.

    This function systematically trains XGBoost models with different combinations of:
    - sample weights
    - scale_pos_weight values

    For each combination, it extracts rules from the fitted models and returns them as a Polars DataFrame.
    The weight transformations loop is parallelized using joblib for improved performance.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weights : list | np.ndarray
        Array of scale_pos_weight values to try. Parallelised across workers.
    sample_weights_df : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names to sample weight arrays.
        If None, uses baseline weights of 1.0 for all samples.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
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
    if (
        len(scale_pos_weights) > len(sample_weights_df.columns)
        if sample_weights_df is not None
        else 1
    ):
        return rule_grid_search_parallel_scales(
            estimator, X_train, y_train, scale_pos_weights, sample_weights_df, n_jobs, verbose
        )
    else:
        return rule_grid_search_parallel_weights(
            estimator, X_train, y_train, scale_pos_weights, sample_weights_df, n_jobs, verbose
        )
