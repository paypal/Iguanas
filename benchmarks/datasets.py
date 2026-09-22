"""Dataset registry and loading.

Datasets are fetched from OpenML via :func:`sklearn.datasets.fetch_openml` and
cached under ``benchmarks/.cache/``.  The registry is deliberately weighted
towards extreme class imbalance (fraud, credit default, churn, defect and
medical screening tasks), which is the regime rule-based alerting systems are
actually deployed in.

Entries whose OpenML coordinates have not been confirmed against a live fetch
carry ``verified=False``.  Use :func:`verify_registry` to check them; nothing
here silently guesses a target column or positive label at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .config import CACHE_DIR

# Floor on retained positives; below roughly this many, folds of the extreme
# imbalance datasets contain too few positives for any model to learn from.
MIN_POSITIVES = 200


@dataclass(frozen=True)
class DatasetSpec:
    """Coordinates and metadata for one binary classification task."""

    name: str
    openml_name: str
    openml_version: int
    target: str
    positive_label: str
    approx_positive_rate: float
    domain: str
    notes: str = ""
    verified: bool = False
    openml_id: int | None = None


@dataclass
class Dataset:
    """A loaded, numerically encoded binary classification task."""

    name: str
    X: pl.DataFrame
    y: np.ndarray
    spec: DatasetSpec
    encoding: dict[str, list[str]] = field(default_factory=dict)

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean())

    @property
    def imbalance_ratio(self) -> float:
        pos = float(self.y.sum())
        return float(len(self.y) - pos) / pos if pos > 0 else float("inf")

    def meta(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "n_rows": int(self.X.height),
            "n_features": int(self.X.width),
            "positive_rate": self.positive_rate,
            "imbalance_ratio": self.imbalance_ratio,
            "domain": self.spec.domain,
            "openml_name": self.spec.openml_name,
            "openml_version": self.spec.openml_version,
            "verified": self.spec.verified,
        }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        # ---- extreme imbalance (< 3% positives) ---------------------------- #
        DatasetSpec(
            "creditcard", "creditcard", 1, "Class", "1", 0.0017, "fraud",
            "ULB credit-card fraud, 284k rows, PCA features.", verified=True,
        ),
        DatasetSpec(
            "aps_failure", "APSFailure", 1, "class", "pos", 0.0167, "industrial",
            "Scania APS failures; heavy missingness.", verified=True,
        ),
        DatasetSpec(
            "pc2", "pc2", 1, "c", "TRUE", 0.0041, "defect",
            "NASA MDP; near-degenerate positive class (23 of 5589).", verified=True,
        ),
        DatasetSpec(
            "mc1", "mc1", 1, "c", "TRUE", 0.0072, "defect",
            "NASA MDP software defect prediction.", verified=True,
        ),
        DatasetSpec(
            "mammography", "mammography", 1, "class", "'1'", 0.0232, "medical",
            "Calcification screening; classic imbalance benchmark.", verified=True,
        ),
        DatasetSpec(
            "satellite_anomaly", "satellite", 1, "Target", "Anomaly", 0.0145, "anomaly",
            "Statlog landsat recast as anomaly detection.", verified=True,
        ),
        # ---- strong imbalance (3-10% positives) ---------------------------- #
        DatasetSpec(
            "wilt", "wilt", 1, "Class", "2", 0.0539, "remote-sensing",
            "Diseased tree detection. OpenML flags v1 inactive.", verified=True,
        ),
        DatasetSpec(
            "sick", "sick", 1, "Class", "sick", 0.0612, "medical",
            "Thyroid screening; small and fast.", verified=True,
        ),
        DatasetSpec(
            "ozone_level_8hr", "ozone-level-8hr", 1, "Class", "2", 0.0631, "environmental",
            "Ozone day detection.", verified=True,
        ),
        DatasetSpec(
            "seismic_bumps", "seismic-bumps", 1, "class", "1", 0.0658, "industrial",
            "UNVERIFIED: name+version 1 resolves to a 210-row 3-class frame, not "
            "the seismic hazard task. Do not use until the coordinates are fixed.",
            verified=False,
        ),
        DatasetSpec(
            "pc1", "pc1", 1, "defects", "true", 0.0694, "defect",
            "NASA MDP software defect prediction.", verified=True,
        ),
        DatasetSpec(
            "pc3", "pc3", 1, "c", "TRUE", 0.1024, "defect",
            "NASA MDP software defect prediction.", verified=True,
        ),
        # ---- moderate imbalance (10-25% positives) ------------------------- #
        DatasetSpec(
            "bank_marketing", "bank-marketing", 1, "Class", "2", 0.1170, "marketing",
            "Term-deposit subscription; churn-like.", verified=True,
        ),
        DatasetSpec(
            "pc4", "pc4", 1, "c", "TRUE", 0.1221, "defect",
            "NASA MDP software defect prediction.", verified=True,
        ),
        DatasetSpec(
            "churn", "churn", 1, "class", "1", 0.1410, "churn",
            "Telecom churn.", verified=True,
        ),
        DatasetSpec(
            "thoracic_surgery", "thoracic-surgery", 1, "Class", "1", 0.1489, "medical",
            "1-year post-operative mortality.", verified=True,
        ),
        DatasetSpec(
            "kc1", "kc1", 1, "defects", "true", 0.1546, "defect",
            "NASA MDP software defect prediction.", verified=True,
        ),
        DatasetSpec(
            "speeddating", "speeddating", 1, "match", "1", 0.1647, "social",
            "Many categorical columns; exercises the encoder.", verified=True,
        ),
        DatasetSpec(
            "jm1", "jm1", 1, "defects", "true", 0.1935, "defect",
            "NASA MDP software defect prediction.", verified=True,
        ),
        DatasetSpec(
            "credit_default", "default-of-credit-card-clients", 1, "y", "1", 0.2212,
            "credit-default", "Taiwan credit-card default.", verified=True,
        ),
        DatasetSpec(
            "hepatitis", "hepatitis", 1, "Class", "DIE", 0.2065, "medical",
            "Very small; useful for degenerate-fold testing.", verified=True,
        ),
        DatasetSpec(
            "blood_transfusion", "blood-transfusion-service-center", 1, "Class", "2",
            0.2380, "medical", "Donor return prediction.", verified=True,
        ),
        DatasetSpec(
            "adult", "adult", 2, "class", ">50K", 0.2393, "census",
            "Mixed categorical/numeric income prediction.", verified=True,
        ),
        # ---- balanced controls -------------------------------------------- #
        DatasetSpec(
            "credit_g", "credit-g", 1, "class", "bad", 0.3000, "credit-default",
            "German credit; small and fast.", verified=True,
        ),
        DatasetSpec(
            "diabetes", "diabetes", 1, "class", "tested_positive", 0.3490, "medical",
            "Pima diabetes; small and fast.", verified=True,
        ),
        DatasetSpec(
            "ionosphere", "ionosphere", 1, "class", "b", 0.3590, "signal",
            "All-numeric control task.", verified=True,
        ),
        DatasetSpec(
            "phoneme", "phoneme", 1, "Class", "2", 0.2935, "speech",
            "All-numeric control task.", verified=True,
        ),
        DatasetSpec(
            "spambase", "spambase", 1, "class", "1", 0.3940, "spam",
            "All-numeric control task.", verified=True,
        ),
    ]
}

SMOKE: tuple[str, ...] = ("diabetes", "credit_g", "sick")


def registry_table() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "name": s.name,
                "openml_name": s.openml_name,
                "openml_version": s.openml_version,
                "target": s.target,
                "positive_label": s.positive_label,
                "approx_positive_rate": s.approx_positive_rate,
                "approx_imbalance_ratio": (1 - s.approx_positive_rate)
                / s.approx_positive_rate,
                "domain": s.domain,
                "verified": s.verified,
                "notes": s.notes,
            }
            for s in REGISTRY.values()
        ]
    ).sort("approx_positive_rate")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class DatasetLoadError(RuntimeError):
    """Raised when a registry entry cannot be materialised as a usable task."""


def _cache_paths(spec: DatasetSpec) -> tuple[Path, Path]:
    stem = f"{spec.name}__v{spec.openml_version}"
    return CACHE_DIR / f"{stem}.parquet", CACHE_DIR / f"{stem}.json"


def _encode_frame(df: pd.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Normalise dtypes only, leaving imputation and encoding to the fold pipeline.

    Numeric columns keep their nulls and categorical columns stay as strings, so
    that :class:`benchmarks.preprocessing.SharedPreprocessor` can learn fill
    values and category levels from the training split alone. Filling here --
    as this function previously did, median-imputing before any split existed --
    leaks test-set statistics, and ordinal-encoding categories invents an
    ordering that a rule learner will happily split on.
    """
    encoding: dict[str, list[str]] = {}
    out: dict[str, pl.Series] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            out[col] = pl.Series(col, series.astype("float64").to_numpy(), dtype=pl.Float64)
        else:
            as_str = series.astype("object").where(series.notna(), None)
            values = [None if v is None else str(v) for v in as_str.tolist()]
            encoding[col] = sorted({v for v in values if v is not None})
            out[col] = pl.Series(col, values, dtype=pl.Utf8)
    return pl.DataFrame(out), encoding


def _binarise_target(raw: pd.Series, spec: DatasetSpec) -> np.ndarray:
    as_str = raw.astype("object").astype(str).str.strip()
    wanted = spec.positive_label.strip().strip("'\"")
    mask = (as_str == wanted) | (as_str == spec.positive_label)
    if not mask.any():
        raise DatasetLoadError(
            f"{spec.name}: positive_label {spec.positive_label!r} not found in target "
            f"{spec.target!r}; observed labels: {sorted(set(as_str))[:12]}"
        )
    if mask.all():
        raise DatasetLoadError(f"{spec.name}: target is constant after binarisation.")
    return mask.to_numpy(dtype=bool)


def _fetch_openml(spec: DatasetSpec) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    kwargs: dict[str, Any] = {"as_frame": True, "parser": "auto"}
    if spec.openml_id is not None:
        bunch = fetch_openml(data_id=spec.openml_id, **kwargs)
    else:
        bunch = fetch_openml(
            name=spec.openml_name, version=spec.openml_version, **kwargs
        )
    frame = bunch.frame
    if frame is None:
        raise DatasetLoadError(f"{spec.name}: OpenML returned no dataframe.")
    return frame


def load_dataset(
    name: str, *, max_rows: int | None = None, seed: int = 0, use_cache: bool = True
) -> Dataset:
    """Load one registry entry, caching the raw frame under ``benchmarks/.cache``.

    Sub-sampling (when ``max_rows`` is set) is stratified and seeded so a run is
    reproducible from the manifest alone.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    frame = _load_raw_frame(spec, use_cache=use_cache)

    if spec.target not in frame.columns:
        raise DatasetLoadError(
            f"{spec.name}: target column {spec.target!r} absent; "
            f"available: {list(frame.columns)[:20]}"
        )
    y = _binarise_target(frame[spec.target], spec)
    X_pd = frame.drop(columns=[spec.target])
    X, encoding = _encode_frame(X_pd)

    if max_rows is not None and X.height > max_rows:
        keep = _stratified_subsample(y, max_rows, seed)
        X = X[keep]
        y = y[keep]
    return Dataset(name=spec.name, X=X, y=y, spec=spec, encoding=encoding)


def _load_raw_frame(spec: DatasetSpec, *, use_cache: bool) -> pd.DataFrame:
    parquet_path, meta_path = _cache_paths(spec)
    if use_cache and parquet_path.exists():
        return pd.read_parquet(parquet_path)

    frame = _fetch_openml(spec)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "name": spec.name,
                "openml_name": spec.openml_name,
                "openml_version": spec.openml_version,
                "n_rows": int(len(frame)),
                "columns": list(map(str, frame.columns)),
            },
            indent=2,
        )
    )
    return frame


def _stratified_subsample(
    y: np.ndarray, max_rows: int, seed: int, *, min_positives: int = MIN_POSITIVES
) -> np.ndarray:
    """Subsample rows while preserving the base rate and the positive class.

    A purely proportional cap annihilates the minority class on the extreme
    imbalance datasets -- capping creditcard (492 positives in 284_807 rows) to
    4_000 rows leaves 7 positives, at which point every model degenerates to
    predicting all-positive. Since the base rate is what these datasets are here
    to exercise, it is held fixed and *max_rows is treated as advisory*: the
    sample grows as needed to retain ``min_positives``.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))

    base_rate = len(pos) / len(y)
    n_pos = min(len(pos), max(int(round(max_rows * base_rate)), min_positives))
    n_neg = min(len(neg), int(round(n_pos * (1.0 - base_rate) / base_rate)))
    keep = np.concatenate(
        [
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ]
    )
    keep.sort()
    return keep


def verify_registry(
    names: list[str] | None = None, *, max_rows: int = 2_000
) -> pl.DataFrame:
    """Attempt to load every (or the given) registry entry and report the outcome.

    This is the only sanctioned way to promote an entry to ``verified=True``:
    run it, read the realised positive rate, then edit the registry by hand.
    """
    targets = names if names is not None else list(REGISTRY)
    rows: list[dict[str, Any]] = []
    for name in targets:
        spec = REGISTRY[name]
        row: dict[str, Any] = {
            "name": name,
            "declared_verified": spec.verified,
            "ok": False,
            "n_rows": None,
            "n_features": None,
            "realised_positive_rate": None,
            "declared_positive_rate": spec.approx_positive_rate,
            "error": "",
        }
        try:
            ds = load_dataset(name, max_rows=max_rows, seed=0)
        except (DatasetLoadError, OSError, ValueError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row.update(
                ok=True,
                n_rows=ds.X.height,
                n_features=ds.X.width,
                realised_positive_rate=ds.positive_rate,
            )
        rows.append(row)
    return pl.DataFrame(rows)
