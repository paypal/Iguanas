import numpy as np
import polars as pl


EPS = 1e-6
_DEFAULT_POWERS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])


def _power_label(power: float) -> str:
    return f"{int(power)}" if power == int(power) else f"{power:.2f}"


def _increasing_exprs(
    col_name: str,
    powers: np.ndarray,
    quantile_val: float | None = None,
    quantile_value: float | None = None,
) -> list[pl.Expr]:
    base = 1 + pl.col(col_name)
    exprs = [pl.lit(1.0).alias("Baseline")]
    for p in powers:
        label = "(1+x)" if p == 1.0 else f"(1+x)^{_power_label(p)}"
        exprs.append(base.pow(p).alias(f"{label}__{col_name}"))
    exprs.append((pl.col(col_name) + 1).log().alias(f"log(1+x)__{col_name}"))
    if quantile_val is not None:
        assert quantile_value is not None
        clipped = 1 + pl.col(col_name).clip(upper_bound=quantile_val)
        q_str = f"{quantile_value * 100:.0f}th"
        for p in powers:
            exprs.append(clipped.pow(p).alias(f"(1+x_clipped_{q_str})^{p:.2f}__{col_name}"))
    return exprs


def _decreasing_exprs(
    col_name: str,
    powers: np.ndarray,
    quantile_val: float | None = None,
    quantile_value: float | None = None,
) -> list[pl.Expr]:
    inv = 1.0 / (pl.col(col_name) + 1.0)
    exprs = []
    for p in powers:
        label = "1/(1+x)" if p == 1.0 else f"1/(1+x)^{_power_label(p)}"
        exprs.append((inv ** p).alias(f"{label}__{col_name}"))
    exprs.append((1.0 / (pl.col(col_name) + 1 + EPS).log()).alias(f"1/log(1+x)__{col_name}"))
    if quantile_val is not None:
        assert quantile_value is not None
        clipped_inv = 1.0 / (pl.col(col_name).clip(upper_bound=quantile_val) + 1.0)
        q_str = f"{quantile_value * 100:.0f}th"
        for p in powers:
            exprs.append(clipped_inv.pow(p).alias(f"1/(1+x_clipped_{q_str})^{p:.2f}__{col_name}"))
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


def generate_increasing_weight(
    X: pl.Series | pl.DataFrame,
    powers: np.ndarray | None = None,
    quantile_value: float | None = None,
) -> pl.DataFrame:
    """Generate weight transformations where larger input values receive larger weights.

    Parameters
    ----------
    X : pl.Series | pl.DataFrame
        Numerical Polars Series or DataFrame to transform. If a DataFrame,
        transformations are applied to each column and concatenated horizontally.
    powers : np.ndarray | None, default=[0.25, 0.5, 1.0, 2.0, 4.0]
        Power values for polynomial transformations.
    quantile_value : float | None, default=None
        If specified, adds clipped transformations at this quantile.

    Returns
    -------
    pl.DataFrame
        Each column is a different weight transformation (baseline, powers, log).
    """
    if (out := _dispatch(generate_increasing_weight, X, powers=powers, quantile_value=quantile_value)) is not None:
        return out
    df, col_name, powers = _resolve(X, powers)
    qval: float | None = float(X.quantile(quantile_value)) if quantile_value is not None else None  # type: ignore[arg-type]
    return df.with_columns(_increasing_exprs(col_name, powers, qval, quantile_value)).drop(col_name)


def generate_decreasing_weight(
    X: pl.Series | pl.DataFrame,
    powers: np.ndarray | None = None,
    quantile_value: float | None = None,
) -> pl.DataFrame:
    """Generate weight transformations where smaller input values receive larger weights.

    Parameters
    ----------
    X : pl.Series | pl.DataFrame
        Numerical Polars Series or DataFrame to transform. If a DataFrame,
        transformations are applied to each column and concatenated horizontally.
    powers : np.ndarray | None, default=[0.25, 0.5, 1.0, 2.0, 4.0]
        Power values for reciprocal transformations (1/(1+x)^power).
    quantile_value : float | None, default=None
        If specified, adds clipped transformations at this quantile.

    Returns
    -------
    pl.DataFrame
        Each column is a different inverse weight transformation.
    """
    if (out := _dispatch(generate_decreasing_weight, X, powers=powers, quantile_value=quantile_value)) is not None:
        return out
    df, col_name, powers = _resolve(X, powers)
    qval: float | None = float(X.quantile(quantile_value)) if quantile_value is not None else None  # type: ignore[arg-type]
    return df.with_columns(
        [pl.lit(1.0).alias("Baseline")] + _decreasing_exprs(col_name, powers, qval, quantile_value)
    ).drop(col_name)


def generate_all_weight(
    X: pl.Series | pl.DataFrame,
    powers: np.ndarray | None = None,
    quantile_value: float | None = None,
) -> pl.DataFrame:
    """Generate all weight transformations (increasing and decreasing).

    Parameters
    ----------
    X : pl.Series | pl.DataFrame
        Numerical Polars Series or DataFrame to transform. If a DataFrame,
        transformations are applied to each column and concatenated horizontally.
    powers : np.ndarray | None, default=[0.25, 0.5, 1.0, 2.0, 4.0]
        Power values used for both increasing and decreasing transformations.
    quantile_value : float | None, default=None
        If specified, adds clipped transformations at this quantile.

    Returns
    -------
    pl.DataFrame
        Combined increasing and decreasing weight transformations in one DataFrame.
    """
    if (out := _dispatch(generate_all_weight, X, powers=powers, quantile_value=quantile_value)) is not None:
        return out
    df, col_name, powers = _resolve(X, powers)
    qval: float | None = float(X.quantile(quantile_value)) if quantile_value is not None else None  # type: ignore[arg-type]
    exprs = _increasing_exprs(col_name, powers, qval, quantile_value)
    exprs += _decreasing_exprs(col_name, powers, qval, quantile_value)
    return df.with_columns(exprs).drop(col_name)

