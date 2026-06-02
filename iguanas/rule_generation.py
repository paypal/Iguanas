from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from xgboost import XGBClassifier

from .rule_formatting import simplify_rule


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
    """Generate metrics for rules extracted from XGBoost trees.

    Parameters
    ----------
    estimator : XGBClassifier
        Fitted XGBoost classifier from which to extract rules.
    all_features_constrained : bool
        If True, uses monotone constraint-based extraction (top-to-bottom).
        If False, uses max gain-based extraction (bottom-to-top).
    **kwargs : dict
        Additional parameters for rule extraction and metric calculation
        (e.g., transformation name, scale_pos_weight value).

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - rule: Extracted rule as a string
        - tree: Tree number from which the rule was extracted
        - scale_pos_weight: Scale_pos_weight value used for this tree
    """
    df = estimator._Booster.trees_to_dataframe()

    rule_strings = []
    tree_ids = []

    # Use groupby for efficient tree iteration
    grouped = df.groupby("Tree", sort=False)

    for tree_id, tree in grouped:
        if tree.empty:
            continue

        if all_features_constrained and isinstance(estimator.monotone_constraints, dict):
            rule = extract_rule_with_monotone_constraints(
                tree, monotone_constraints=estimator.monotone_constraints
            )
            rule = simplify_rule(rule)
        else:
            rule = extract_rule_by_max_gain(tree)
            rule = simplify_rule(rule)

        if not rule:
            continue

        rule_strings.append(rule)
        tree_ids.append(tree_id)

    # Create single DataFrame at the end
    if rule_strings:
        rules_data = {"rule": rule_strings, "tree": tree_ids}
        # Add kwargs columns
        for key, value in kwargs.items():
            rules_data[key] = [value] * len(rule_strings)
        return pd.DataFrame(rules_data)
    return pd.DataFrame()


def _check_all_features_have_monotone_constraints(
    estimator: XGBClassifier, n_features: int
) -> bool:
    """
    Check if all features have non-zero monotone constraints.

    Parameters
    ----------
    estimator : XGBClassifier
        The estimator to check
    n_features : int
        Expected number of features

    Returns
    -------
    bool
        True if all features have constraints of +1 or -1
    """
    if not estimator.monotone_constraints:
        return False
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
    scale_pos_weight_vec: np.ndarray,
    all_features_constrained: bool,
    feature_names: list[str] | None = None,
) -> list[pd.DataFrame]:
    """
    Process a single weight column across all scale_pos_weight values.

    This helper function is used for parallel execution in rule_grid_search.

    Parameters
    ----------
    weights : pd.Series | np.ndarray
        Weights for this transformation
    estimator_params : dict
        XGBoost estimator parameters to reconstruct the model
    X_train : pd.DataFrame | np.ndarray
        Training features as numpy array (serializes faster than DataFrame for IPC).
    y_train : pd.Series | np.ndarray
        Training target as numpy array.
    scale_pos_weight_vec : np.ndarray
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
    rules_vec = []
    transformation = weights.name if hasattr(weights, "name") else "Baseline"  # type: ignore
    weights_array = weights.values if hasattr(weights, "values") else weights  # type: ignore

    # Reconstruct DataFrame from numpy + names so XGBoost preserves feature names
    # in the booster (needed for monotone-constraint extraction and readable rules).
    # This is cheap — the array is already deserialized; only metadata is created.
    if feature_names is not None and isinstance(X_train, np.ndarray):
        X_fit: pd.DataFrame | np.ndarray = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_fit = X_train

    for scale_pos_weight in scale_pos_weight_vec:
        est = XGBClassifier(**estimator_params)
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
            rules_vec.append(rules_df)

    return rules_vec


def _train_rules_for_scale(
    scale_pos_weight: float,
    weights_np: np.ndarray,
    weight_names: list[str],
    estimator_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    all_features_constrained: bool,
    feature_names: list[str] | None = None,
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
    weight_names : list[str]
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
    rules_vec = []

    if feature_names is not None and isinstance(X_train, np.ndarray):
        X_fit: pd.DataFrame | np.ndarray = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_fit = X_train
    for i, name in enumerate(weight_names):
        weights_array = weights_np[:, i]
        est = XGBClassifier(**estimator_params)
        est.set_params(scale_pos_weight=scale_pos_weight)
        try:
            est.fit(X_fit, y_train, sample_weight=weights_array)
        except Exception:
            continue

        params = {
            "transformation": name,
            "scale_pos_weight": scale_pos_weight,
        }
        rules_df = extract_rules(est, all_features_constrained, **params)
        if not rules_df.empty:
            rules_vec.append(rules_df)

    return rules_vec


def rule_grid_search_sequential(
    estimator: XGBClassifier,
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weight_vec: list[float] | np.ndarray,
    weights_train_vec: pl.DataFrame | pd.DataFrame | None = None,
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
    scale_pos_weight_vec : list | np.ndarray
        Array of scale_pos_weight values to try.
    weights_train_vec : pl.DataFrame | pd.DataFrame | None, default=None
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
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    feature_names = X_train.columns if isinstance(X_train, pl.DataFrame) else list(X_train.columns)

    if X_train_np.dtype == object:
        raise ValueError(
            "X_train contains non-numeric data. Please encode categorical features "
            "numerically before using rule_grid_search_parallel_scales."
        )

    if len(scale_pos_weight_vec) == 0:
        raise ValueError("scale_pos_weight_vec cannot be empty")

    if weights_train_vec is None:
        weights_train_vec_pd = pd.DataFrame({"Baseline": np.ones(len(X_train))})
    elif isinstance(weights_train_vec, pl.DataFrame):
        weights_train_vec_pd = weights_train_vec.to_pandas()
    else:
        weights_train_vec_pd = weights_train_vec

    weight_names = list(weights_train_vec_pd.columns)
    weights_np = weights_train_vec_pd.to_numpy()
    estimator_params = estimator.get_params()
    estimator_params.pop("scale_pos_weight", None)

    n_features = len(X_train.columns)
    all_features_constrained = _check_all_features_have_monotone_constraints(estimator, n_features)

    if verbose > 0:
        print(
            f"Starting sequential rule grid search with {len(weight_names)} weight "
            f"transformations and {len(scale_pos_weight_vec)} scale_pos_weight values "
            f"({len(weight_names) * len(scale_pos_weight_vec)} total combinations)"
        )

    rules_vec = []
    for scale_pos_weight in scale_pos_weight_vec:
        results = _train_rules_for_scale(
            scale_pos_weight,
            weights_np,
            weight_names,
            estimator_params,
            X_train_np,
            y_train_np,
            all_features_constrained,
            feature_names=feature_names,
        )
        rules_vec.extend(results)

    if rules_vec:
        final_X = pl.from_pandas(pd.concat(rules_vec, ignore_index=True))
    else:
        final_X = pl.DataFrame()

    final_X = final_X.unique("rule") if final_X.height > 0 else final_X
    if verbose > 0:
        print(f"Extracted {len(final_X)} total rules from sequential grid search")

    return final_X


def rule_grid_search_parallel_weights(
    estimator: XGBClassifier,
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weight_vec: list | np.ndarray,
    weights_train_vec: pl.DataFrame | pd.DataFrame | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> pl.DataFrame:
    """
    Perform grid search over sample weight transformations and scale_pos_weight values to find optimal rules.

    This function systematically trains XGBoost models with different combinations of:
    - sample weight transformations (e.g., linear, power, logarithmic, clipped)
    - scale_pos_weight values to handle class imbalance

    For each combination, it extracts rules from the fitted models and returns them as a Polars DataFrame.
    The weight transformations loop is parallelized using joblib for improved performance.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction. The estimator's
        hyperparameters (except scale_pos_weight and sample_weight) will be used as defaults
        for training multiple models during the grid search.
    X_train : pl.DataFrame | pd.DataFrame | np.ndarray
        Training feature matrix (without target column). Can be Polars/Pandas DataFrame or NumPy array.
        NumPy arrays provide fastest serialization for parallel processing.
    y_train : pl.Series | pd.Series | np.ndarray
        Training target values
    scale_pos_weight_vec : list | np.ndarray
        Array of scale_pos_weight values to try
    weights_train_vec : pl.DataFrame | pd.DataFrame | None, default=None
        DataFrame mapping transformation names (columns) to sample weight arrays (rows).
        If None, uses baseline weights of 1.0 for all samples.
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors
    verbose : int, default=0
        Controls the verbosity level. Higher values show more information:

        - 0: silent (no output)
        - 1: progress information (start/end summary)
        - >=2: detailed progress with live updates from joblib Parallel backend

    Returns
    -------
    pl.DataFrame
        DataFrame containing rule information and metrics for all combinations of
        weight transformations and scale_pos_weight values. Each row includes:

        - rule: str, the extracted rule as a string
        - tree: int, the tree number from which the rule was extracted
        - scale_pos_weight: float, the scale_pos_weight value used for this tree
        - transformation: str, the name of the weight transformation used

    Examples
    --------
    >>> weights_train = generate_sample_weight_transformations(X_train["amount"])
    >>> scale_weights = np.logspace(0, np.log10(imbalance_ratio*2), 20)
    >>> results = rule_grid_search(
    ...     estimator, X_train, y_train,
    ...     scale_weights, weights_train, n_jobs=-1, verbose=1
    ... )
    """
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    feature_names = X_train.columns if isinstance(X_train, pl.DataFrame) else list(X_train.columns)

    if X_train_np.dtype == object:
        raise ValueError(
            "X_train contains non-numeric data. Please encode categorical features "
            "numerically before using rule_grid_search_parallel_scales."
        )

    if len(scale_pos_weight_vec) == 0:
        raise ValueError("scale_pos_weight_vec cannot be empty")

    if weights_train_vec is None:
        weights_train_vec_pd = pd.DataFrame({"Baseline": np.ones(len(X_train))})
    elif isinstance(weights_train_vec, pl.DataFrame):
        weights_train_vec_pd = weights_train_vec.to_pandas()
    else:
        weights_train_vec_pd = weights_train_vec

    weight_columns = weights_train_vec_pd.columns
    estimator_params = estimator.get_params()
    estimator_params.pop("scale_pos_weight", None)

    n_features = len(X_train.columns)
    all_features_constrained = _check_all_features_have_monotone_constraints(estimator, n_features)

    if verbose > 0:
        print(
            f"Starting rule grid search with {len(weight_columns)} weight transformations "
            f"and {len(scale_pos_weight_vec)} scale_pos_weight values "
            f"({len(weight_columns) * len(scale_pos_weight_vec)} total combinations)"
        )

    # Map verbose levels for joblib: 0=silent, 1=silent, 2+=detailed (10+)
    joblib_verbose = 10 if verbose >= 2 else 0

    results_nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=joblib_verbose)(
        delayed(_train_rules_for_weight_transformation)(
            weights_train_vec_pd[name],
            estimator_params,
            X_train_np,
            y_train_np,
            scale_pos_weight_vec,
            all_features_constrained,
            feature_names,
        )
        for name in weight_columns
    )

    rules_vec = []
    for sublist in results_nested:
        if sublist is not None:
            rules_vec.extend(sublist)

    if rules_vec:
        final_X_pd = pd.concat(rules_vec, ignore_index=True)
        final_X = pl.from_pandas(final_X_pd)
    else:
        final_X = pl.DataFrame()

    final_X = final_X.unique("rule") if final_X.height > 0 else final_X
    if verbose > 0:
        print(f"Extracted {len(final_X)} total rules from grid search")

    return final_X


def rule_grid_search_parallel_scales(
    estimator: XGBClassifier,
    X_train: pl.DataFrame | pd.DataFrame,
    y_train: pl.Series | pd.Series,
    scale_pos_weight_vec: list | np.ndarray,
    weights_train_vec: pl.DataFrame | pd.DataFrame | None = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> pl.DataFrame:
    """
    Perform grid search parallelised over scale_pos_weight values.

    Identical behaviour to :func:`rule_grid_search_parallel_weights` but parallelises
    over the ``scale_pos_weight_vec`` axis instead of the weight-transformation axis.
    Prefer this variant when ``len(scale_pos_weight_vec) >= len(weights_train_vec.columns)``
    so that workers are kept maximally busy.

    Parameters
    ----------
    estimator : XGBClassifier
        Base XGBoost classifier to use as a template for rule extraction.
    X_train : pl.DataFrame | pd.DataFrame
        Training feature matrix.
    y_train : pl.Series | pd.Series
        Training target values.
    scale_pos_weight_vec : list | np.ndarray
        Array of scale_pos_weight values to try. Parallelised across workers.
    weights_train_vec : pl.DataFrame | pd.DataFrame | None, default=None
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
        Same schema as :func:`rule_grid_search_parallel_weights`: columns rule, tree,
        scale_pos_weight, transformation.
    """
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    feature_names = X_train.columns if isinstance(X_train, pl.DataFrame) else list(X_train.columns)

    if X_train_np.dtype == object:
        raise ValueError(
            "X_train contains non-numeric data. Please encode categorical features "
            "numerically before using rule_grid_search_parallel_scales."
        )
    if len(scale_pos_weight_vec) == 0:
        raise ValueError("scale_pos_weight_vec cannot be empty")

    if weights_train_vec is None:
        weights_train_vec_pd = pd.DataFrame({"Baseline": np.ones(len(X_train))})
    elif isinstance(weights_train_vec, pl.DataFrame):
        weights_train_vec_pd = weights_train_vec.to_pandas()
    else:
        weights_train_vec_pd = weights_train_vec

    weight_names = list(weights_train_vec_pd.columns)
    weights_np = weights_train_vec_pd.to_numpy()
    estimator_params = estimator.get_params()
    estimator_params.pop("scale_pos_weight", None)

    n_features = len(X_train.columns)
    all_features_constrained = _check_all_features_have_monotone_constraints(estimator, n_features)

    if verbose > 0:
        print(
            f"Starting parallel-scales rule grid search with {len(weight_names)} weight "
            f"transformations and {len(scale_pos_weight_vec)} scale_pos_weight values "
            f"({len(weight_names) * len(scale_pos_weight_vec)} total combinations)"
        )

    joblib_verbose = 10 if verbose >= 2 else 0

    results_nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=joblib_verbose)(
        delayed(_train_rules_for_scale)(
            scale_pos_weight,
            weights_np,
            weight_names,
            estimator_params,
            X_train_np,
            y_train_np,
            all_features_constrained,
            feature_names,
        )
        for scale_pos_weight in scale_pos_weight_vec
    )

    rules_vec = []
    for sublist in results_nested:
        if sublist is not None:
            rules_vec.extend(sublist)

    if rules_vec:
        final_X_pd = pd.concat(rules_vec, ignore_index=True)
        final_X = pl.from_pandas(final_X_pd)
    else:
        final_X = pl.DataFrame()

    final_X = final_X.unique("rule") if final_X.height > 0 else final_X
    if verbose > 0:
        print(f"Extracted {len(final_X)} total rules from parallel-scales grid search")

    return final_X
