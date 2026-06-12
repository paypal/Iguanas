import numpy as np
import polars as pl

from .rule_selection import filter_correlated_rules

EPS = 1e-6
_DEFAULT_POWERS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])


def _power_label(power: float) -> str:
    return f"{int(power)}" if power == int(power) else f"{power:.2f}"


def _increasing_exprs(
    col_name: str,
    powers: np.ndarray,
) -> list[pl.Expr]:
    base = 1 + pl.col(col_name)
    exprs = [pl.lit(1.0).alias("Baseline")]
    for p in powers:
        label = "(1+x)" if p == 1.0 else f"(1+x)^{_power_label(p)}"
        exprs.append(base.pow(p).alias(f"{label}__{col_name}"))
    exprs.append((pl.col(col_name) + 1).log().alias(f"log(1+x)__{col_name}"))
    return exprs


def _decreasing_exprs(
    col_name: str,
    powers: np.ndarray,
) -> list[pl.Expr]:
    inv = 1.0 / (pl.col(col_name) + 1.0)
    exprs = []
    for p in powers:
        label = "1/(1+x)" if p == 1.0 else f"1/(1+x)^{_power_label(p)}"
        exprs.append((inv**p).alias(f"{label}__{col_name}"))
    exprs.append((1.0 / (pl.col(col_name) + 1 + EPS).log()).alias(f"1/log(1+x)__{col_name}"))
    return exprs


def _resolve(
    X: pl.Series | pl.DataFrame, powers: np.ndarray | None
) -> tuple[pl.DataFrame, str, np.ndarray]:
    assert isinstance(X, pl.Series)
    powers = _DEFAULT_POWERS if powers is None else np.asarray(powers)
    col_name = X.name
    return pl.DataFrame({col_name: X - X.min()}), col_name, powers


def _dispatch(fn, X: pl.Series | pl.DataFrame, **kwargs) -> pl.DataFrame | None:
    """Handle pl.DataFrame input by applying fn per column and concatenating."""
    if not isinstance(X, pl.DataFrame):
        return None
    results = [fn(X[c], **kwargs) for c in X.columns]
    return pl.concat([results[0]] + [r.drop("Baseline") for r in results[1:]], how="horizontal")


def generate_increasing_weights(
    X: pl.Series | pl.DataFrame,
    powers: np.ndarray | None = None,
) -> pl.DataFrame:
    """Generate weight transformations where larger input values receive larger weights.

    Parameters
    ----------
    X : pl.Series | pl.DataFrame
        Numerical Polars Series or DataFrame to transform. If a DataFrame,
        transformations are applied to each column and concatenated horizontally.
    powers : np.ndarray | None, default=[0.25, 0.5, 1.0, 2.0, 4.0]
        Power values for polynomial transformations.

    Returns
    -------
    pl.DataFrame
        Each column is a different weight transformation (baseline, powers, log).

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series("amount", [0.0, 10.0, 50.0, 100.0])
    >>> df = generate_increasing_weights(s)
    >>> df.columns  # 'Baseline', '(1+x)^0.25__amount', ..., 'log(1+x)__amount'
    >>> # DataFrame input: each column processed independently
    >>> X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    >>> generate_increasing_weights(X).shape
    (3, ...)  # 1 Baseline + 5 power cols + 1 log col per feature, minus duplicate Baselines
    """
    if (out := _dispatch(generate_increasing_weights, X, powers=powers)) is not None:
        return out
    df, col_name, powers = _resolve(X, powers)
    return df.with_columns(_increasing_exprs(col_name, powers)).drop(col_name)


def generate_decreasing_weights(
    X: pl.Series | pl.DataFrame,
    powers: np.ndarray | None = None,
) -> pl.DataFrame:
    """Generate weight transformations where smaller input values receive larger weights.

    Parameters
    ----------
    X : pl.Series | pl.DataFrame
        Numerical Polars Series or DataFrame to transform. If a DataFrame,
        transformations are applied to each column and concatenated horizontally.
    powers : np.ndarray | None, default=[0.25, 0.5, 1.0, 2.0, 4.0]
        Power values for reciprocal transformations (1/(1+x)^power).

    Returns
    -------
    pl.DataFrame
        Each column is a different inverse weight transformation.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series("amount", [0.0, 10.0, 50.0, 100.0])
    >>> df = generate_decreasing_weights(s)
    >>> df.columns  # '1/(1+x)__amount', '1/(1+x)^0.25__amount', ..., '1/log(1+x)__amount'
    """
    if (out := _dispatch(generate_decreasing_weights, X, powers=powers)) is not None:
        return out
    df, col_name, powers = _resolve(X, powers)
    return df.with_columns(
        [pl.lit(1.0).alias("Baseline")] + _decreasing_exprs(col_name, powers)
    ).drop(col_name)


def generate_weights(
    X: pl.Series | pl.DataFrame,
    powers: np.ndarray | None = None,
) -> pl.DataFrame:
    """Generate all weight transformations (increasing and decreasing).

    Parameters
    ----------
    X : pl.Series | pl.DataFrame
        Numerical Polars Series or DataFrame to transform. If a DataFrame,
        transformations are applied to each column and concatenated horizontally.
    powers : np.ndarray | None, default=[0.25, 0.5, 1.0, 2.0, 4.0]
        Power values used for both increasing and decreasing transformations.

    Returns
    -------
    pl.DataFrame
        Combined increasing and decreasing weight transformations in one DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series("amount", [0.0, 10.0, 50.0, 100.0])
    >>> df = generate_weights(s)
    >>> # Columns include both (1+x)^p and 1/(1+x)^p families plus log variants
    """
    if (out := _dispatch(generate_weights, X, powers=powers)) is not None:
        return out
    df, col_name, powers = _resolve(X, powers)
    exprs = _increasing_exprs(col_name, powers)
    exprs += _decreasing_exprs(col_name, powers)
    return df.with_columns(exprs).drop(col_name)


def select_uncorrelated_weights(
    sample_weights_df: pl.DataFrame,
    importance: dict[str, float],
    target_len: int,
    min_corr: float = 0.01,
    max_corr: float = 0.99,
    step: float = 0.01,
    use_abs: bool = False,
) -> tuple[list[str], float]:
    """Return a filtered set of weight columns closest to a target length.

    The function searches the correlation threshold range ``[min_corr, max_corr]``
    (discretised by ``step``) using binary search. For each candidate threshold
    it calls :func:`iguanas.rule_selection.filter_correlated_rules` with
    ``max_corr`` set to the candidate value and returns the first filtered list
    whose length is ``>= target_len``.

    Parameters
    ----------
    sample_weights_df : pl.DataFrame
        DataFrame containing candidate weight series (columns are weight names).
    importance : dict[str, float]
        Mapping from rule/weight name to importance score used by the filter.
    target_len : int
        Desired number of selected rules (must be non-negative).
    min_corr : float, default=0.01
        Minimum correlation threshold to consider (lower bound of search).
    max_corr : float, default=0.99
        Maximum correlation threshold to consider (upper bound of search).
    step : float, default=0.01
        Step size used to discretise the correlation thresholds in the search.
    use_abs : bool, default=False
        If True, use absolute correlation values when filtering.

    Returns
    -------
    tuple[list[str], float]
        A tuple ``(filtered_names, corr_value)`` where ``filtered_names`` is the
        list of selected weight names at the chosen correlation threshold and
        ``corr_value`` is the correlation threshold that produced that list.

    Notes
    -----
    If ``target_len`` is below the minimum achievable length at ``min_corr``,
    the minimum result is returned. If it is above the maximum achievable length
    at ``max_corr``, the maximum result is returned. The search discretises
    thresholds as ``i * step`` where ``i`` ranges between ``round(min_corr/step)``
    and ``round(max_corr/step)``.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"w1": [0.1, 0.2], "w2": [0.0, 0.3]})
    >>> selected, corr = select_uncorrelated_weights(df, {"w1": 1.0, "w2": 0.5}, 1)
    """
    if target_len < 0:
        raise ValueError("target_len must be non-negative")
    if not 0 < min_corr < max_corr < 1.0:
        raise ValueError("min_corr and max_corr must satisfy 0 < min_corr < max_corr < 1.0")
    if step <= 0:
        raise ValueError("step must be positive")

    min_step = int(round(min_corr / step))
    max_step = int(round(max_corr / step))

    def compute_filtered(step_idx: int) -> tuple[int, list[str], float]:
        max_corr_value = step_idx * step
        filtered = filter_correlated_rules(
            sample_weights_df,
            importance=importance,
            max_corr=max_corr_value,
            use_abs=use_abs,
        )
        return len(filtered), filtered, max_corr_value

    min_len, min_filtered, min_corr_value = compute_filtered(min_step)
    max_len, max_filtered, max_corr_value = compute_filtered(max_step)

    if target_len <= min_len:
        return min_filtered, min_corr_value
    if target_len >= max_len:
        return max_filtered, max_corr_value

    lo = min_step
    hi = max_step
    while lo <= hi:
        mid = (lo + hi) // 2
        cur_len, cur_filtered, cur_corr_value = compute_filtered(mid)

        if cur_len == target_len:
            return cur_filtered, cur_corr_value
        if cur_len < target_len:
            lo = mid + 1
        else:
            hi = mid - 1

    if lo > max_step:
        return max_filtered, max_corr_value

    _, upper_filtered, upper_corr_value = compute_filtered(lo)
    return upper_filtered, upper_corr_value
