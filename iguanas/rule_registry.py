from __future__ import annotations

import builtins
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


class RuleRegistry:
    """Store, version, and compare named rule snapshots.

    Each snapshot records the rule list, optional metrics, optional metadata,
    and a UTC timestamp.  The registry can be persisted to/from a JSON file
    for cross-session use, or kept in-memory only.

    Parameters
    ----------
    path : str | Path | None, default=None
        Path to a JSON file used for persistence.  If the file already exists
        it is loaded on construction.  If ``None``, the registry is in-memory
        only and snapshots are lost when the object is garbage-collected.

    Examples
    --------
    >>> registry = RuleRegistry("rules.json")
    >>> registry.save("v1", rules=['(X["age"] > 30)'])
    >>> registry.list()
    ['v1']
    >>> entry = registry.load("v1")
    >>> entry["rules"]
    ['(X["age"] > 30)']
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path: Path | None = Path(path) if path is not None else None
        self._registry: dict[str, dict[str, Any]] = {}
        if self._path is not None and self._path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        name: str,
        rules: list[str],
        metrics: pl.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a named ruleset snapshot.

        Overwrites any existing snapshot with the same name.

        Parameters
        ----------
        name : str
            Snapshot identifier.
        rules : list[str]
            Rule expression strings to store.
        metrics : pl.DataFrame | None, default=None
            Optional metrics DataFrame (e.g. from :func:`~iguanas.metrics.compute_metrics`).
            Stored internally as a list of dicts for JSON compatibility.
        metadata : dict | None, default=None
            Arbitrary key-value metadata (e.g. threshold settings, dataset
            description, experiment notes).
        """
        self._registry[name] = {
            "rules": rules,
            "metrics": metrics.to_dicts() if metrics is not None else None,
            "metadata": metadata or {},
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if self._path is not None:
            self._dump_to_disk()

    def load(self, name: str) -> dict[str, Any]:
        """Load a saved snapshot by name.

        Parameters
        ----------
        name : str
            Snapshot name to retrieve.

        Returns
        -------
        dict
            Dict with keys:

            - ``"rules"`` — list of rule expression strings
            - ``"metrics"`` — ``pl.DataFrame`` if metrics were saved, else ``None``
            - ``"metadata"`` — dict of metadata
            - ``"saved_at"`` — ISO-8601 UTC timestamp string

        Raises
        ------
        KeyError
            If no snapshot with the given name exists.
        """
        if name not in self._registry:
            raise KeyError(f"No snapshot named {name!r}. Available: {self.list()}")
        entry = self._registry[name].copy()
        if entry["metrics"] is not None:
            entry["metrics"] = pl.DataFrame(entry["metrics"])
        return entry

    def list(self) -> list[str]:
        """Return a sorted list of all snapshot names."""
        return sorted(self._registry)

    def delete(self, name: str) -> None:
        """Delete a snapshot by name.

        Parameters
        ----------
        name : str
            Snapshot to remove.

        Raises
        ------
        KeyError
            If no snapshot with the given name exists.
        """
        if name not in self._registry:
            raise KeyError(f"No snapshot named {name!r}. Available: {self.list()}")
        del self._registry[name]
        if self._path is not None:
            self._dump_to_disk()

    def compare(
        self,
        name_a: str,
        name_b: str,
        metric_cols: builtins.list[str] | None = None,
    ) -> pl.DataFrame:
        """Compare saved metrics of two snapshots side by side.

        Parameters
        ----------
        name_a : str
            Name of the first snapshot.
        name_b : str
            Name of the second snapshot.
        metric_cols : list[str] | None, default=None
            Metric columns to include.  When ``None``, all columns present
            in both snapshots (excluding ``"rule"``) are used.

        Returns
        -------
        pl.DataFrame
            One row per rule found in either snapshot.  Metric columns are
            suffixed with ``_{name_a}`` and ``_{name_b}``.  Rules absent
            from one snapshot produce null values for its columns.

        Raises
        ------
        ValueError
            If either snapshot was saved without a metrics DataFrame.
        KeyError
            If either snapshot name does not exist.
        """
        entry_a = self.load(name_a)
        entry_b = self.load(name_b)

        if entry_a["metrics"] is None:
            raise ValueError(f"Snapshot {name_a!r} has no saved metrics.")
        if entry_b["metrics"] is None:
            raise ValueError(f"Snapshot {name_b!r} has no saved metrics.")

        m_a: pl.DataFrame = entry_a["metrics"]
        m_b: pl.DataFrame = entry_b["metrics"]

        if metric_cols is None:
            metric_cols = [c for c in m_a.columns if c != "rule" and c in m_b.columns]

        m_a = m_a.select(["rule"] + [c for c in metric_cols if c in m_a.columns])
        m_b = m_b.select(["rule"] + [c for c in metric_cols if c in m_b.columns])

        m_a = m_a.rename({c: f"{c}_{name_a}" for c in m_a.columns if c != "rule"})
        m_b = m_b.rename({c: f"{c}_{name_b}" for c in m_b.columns if c != "rule"})

        return m_a.join(m_b, on="rule", how="full", coalesce=True).sort("rule")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _dump_to_disk(self) -> None:
        assert self._path is not None
        self._path.write_text(json.dumps(self._registry, indent=2, default=str))

    def _load_from_disk(self) -> None:
        assert self._path is not None
        self._registry = json.loads(self._path.read_text())


def filter_rule_pairs_by_overlap(
    R: pl.DataFrame,
    min_overlap: float = 0.0,
    max_overlap: float = 1.0,
) -> pl.DataFrame:
    """Return rule pairs whose Jaccard overlap falls within ``[min_overlap, max_overlap]``.

    Jaccard similarity is defined as
    *(samples flagged by both) / (samples flagged by either)*.
    The two bounds let you slice any region of the overlap spectrum:

    - **Disjoint pairs** (never co-fire): ``max_overlap=0.0``
    - **Near-disjoint pairs**: ``max_overlap=0.1``
    - **All pairs**: defaults ``min_overlap=0.0, max_overlap=1.0``
    - **Redundant pairs** (near-identical): ``min_overlap=0.9``

    Parameters
    ----------
    R : pl.DataFrame
        Boolean DataFrame of rule predictions (columns = rules, rows = samples).
    min_overlap : float, default=0.0
        Lower Jaccard bound (inclusive).  Pairs with ``jaccard < min_overlap``
        are excluded.
    max_overlap : float, default=1.0
        Upper Jaccard bound (inclusive).  Pairs with ``jaccard > max_overlap``
        are excluded.

    Returns
    -------
    pl.DataFrame
        Matching rule pairs with columns:
        ``rule_a``, ``rule_b``, ``jaccard``, ``flagged_by_both``,
        ``flagged_by_either``.  Sorted by ``jaccard`` ascending.
        Returns an empty DataFrame with the correct schema when no pairs match.

    Examples
    --------
    >>> import polars as pl
    >>> R = pl.DataFrame({
    ...     "rule_A": [True,  True,  False, False],
    ...     "rule_B": [False, False, True,  True],  # disjoint from rule_A
    ...     "rule_C": [True,  False, True,  False],
    ... })
    >>> filter_rule_pairs_by_overlap(R, max_overlap=0.0)   # only disjoint pairs
    shape: (1, 5)  # rule_A vs rule_B
    >>> filter_rule_pairs_by_overlap(R, min_overlap=0.3)   # only overlapping pairs
    """
    _EMPTY_SCHEMA = {
        "rule_a": pl.String,
        "rule_b": pl.String,
        "jaccard": pl.Float64,
        "flagged_by_both": pl.Int64,
        "flagged_by_either": pl.Int64,
    }

    rules = R.columns
    if len(rules) < 2:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    records: list[dict[str, Any]] = []
    for i, rule_a in enumerate(rules):
        for rule_b in rules[i + 1 :]:
            a = R[rule_a]
            b = R[rule_b]
            both = int((a & b).sum())
            either = int((a | b).sum())
            jaccard = both / either if either > 0 else 0.0
            if min_overlap <= jaccard <= max_overlap:
                records.append(
                    {
                        "rule_a": rule_a,
                        "rule_b": rule_b,
                        "jaccard": jaccard,
                        "flagged_by_both": both,
                        "flagged_by_either": either,
                    }
                )

    if not records:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    return pl.DataFrame(records).sort("jaccard")
