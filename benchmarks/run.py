"""CLI entrypoint for the Iguanas benchmark harness.

    python -m benchmarks.run --smoke
    python -m benchmarks.run --full --seeds 5
    python -m benchmarks.run --verify-registry
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from . import __version__
from .ablations import run_ablations
from .baselines import Unavailable, availability_report, make_baseline
from .config import CACHE_DIR, FULL_CONFIG, SMOKE_CONFIG, ExperimentConfig
from .datasets import REGISTRY, SMOKE, Dataset, DatasetLoadError, load_dataset, registry_table, verify_registry
from .iguanas_adapter import IguanasAdapter
from .protocol import FoldResult, make_nested_splits, prepare_fold, run_fold
from .timing import FitTimeout, time_limit, timed
from .reporting import build_report, write_table
from .timing import hardware_info

_PACKAGES = (
    "iguanas", "numpy", "pandas", "polars", "scikit-learn", "scipy",
    "xgboost", "imodels", "wittgenstein", "scikit-posthocs", "openml",
)


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {type(exc).__name__}"
    return completed.stdout.strip()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def build_manifest(cfg: ExperimentConfig, datasets: list[str], mode: str) -> dict[str, Any]:
    return {
        "harness_version": __version__,
        "mode": mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "argv": sys.argv,
        "datasets": datasets,
        "config": _jsonable(cfg),
        "packages": _package_versions(),
        "hardware": hardware_info(),
    }


def live_model_names(cfg: ExperimentConfig) -> list[str]:
    """Names of the models that can actually be constructed in this environment."""
    names = ["iguanas"]
    names += [n for n in cfg.baselines if not isinstance(make_baseline(n, cfg, 0), Unavailable)]
    return names


def _new_model(name: str, cfg: ExperimentConfig, seed: int) -> Any:
    """A fresh, unfitted model instance; never reuse one across folds."""
    if name == "iguanas":
        return IguanasAdapter(cfg, seed=seed)
    model = make_baseline(name, cfg, seed)
    if isinstance(model, Unavailable):
        raise RuntimeError(f"{name} became unavailable mid-run: {model.reason}")
    return model


def run_benchmark(
    cfg: ExperimentConfig,
    dataset_names: list[str],
    *,
    verbose: bool = True,
    checkpoint_dir: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run every model over every dataset under the nested protocol.

    When *checkpoint_dir* is given, partial results are flushed after each
    dataset so a long run interrupted part-way still yields usable data.
    """
    rows: list[dict[str, Any]] = []
    dataset_meta: list[dict[str, Any]] = []
    for name in dataset_names:
        try:
            dataset = load_dataset(name, max_rows=cfg.max_rows, seed=cfg.seeds[0])
        except (DatasetLoadError, OSError, ValueError, KeyError) as exc:
            if verbose:
                print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            dataset_meta.append({"dataset": name, "loaded": False, "error": str(exc)})
            continue
        dataset_meta.append({**dataset.meta(), "loaded": True, "error": ""})
        if verbose:
            print(
                f"  {name}: {dataset.X.height} rows x {dataset.X.width} features, "
                f"positive rate {dataset.positive_rate:.4f}"
            )
        started = time.perf_counter()
        # A single misbehaving dataset must not destroy a multi-hour run over
        # all the others; record it and move on.
        try:
            rows.extend(_run_dataset(dataset, cfg, verbose=verbose))
        except Exception as exc:  # noqa: BLE001 - isolation boundary
            dataset_meta[-1]["error"] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"  [abort] {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if verbose:
            print(f"  {name}: finished in {time.perf_counter() - started:.1f}s", flush=True)
            _print_dataset_summary(rows, name, cfg.primary_alert_rate)
        if checkpoint_dir is not None and rows:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rows, infer_schema_length=None).write_csv(
                checkpoint_dir / "raw_results.partial.csv"
            )
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(dataset_meta)


def _print_dataset_summary(
    rows: list[dict[str, Any]], dataset_name: str, alert_rate: float
) -> None:
    """Per-model mean precision/recall/f1/complexity at one dataset, one budget."""
    subset = [
        r
        for r in rows
        if r["dataset"] == dataset_name
        and r["status"] == "ok"
        and r["target_alert_rate"] == alert_rate
    ]
    if not subset:
        print(f"    (no successful runs for {dataset_name} at ar={alert_rate})")
        return
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in subset:
        by_model.setdefault(r["model"], []).append(r)
    print(f"    --- {dataset_name} @ alert_rate={alert_rate} ---")
    print(f"    {'model':<14} {'n':>3} {'precision':>10} {'recall':>8} {'f1':>8} {'conditions':>11}")
    for model, rs in sorted(by_model.items(), key=lambda kv: -sum(x["test_f1"] for x in kv[1]) / len(kv[1])):
        n = len(rs)
        prec = sum(r["test_precision"] for r in rs) / n
        rec = sum(r["test_recall"] for r in rs) / n
        f1 = sum(r["test_f1"] for r in rs) / n
        cond = sum(r["complexity_conditions"] for r in rs) / n
        print(f"    {model:<14} {n:>3} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f} {cond:>11.1f}")


def _run_dataset(
    dataset: Dataset, cfg: ExperimentConfig, *, verbose: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = live_model_names(cfg)
    for seed in cfg.seeds:
        splits = make_nested_splits(dataset.y, cfg, seed)
        for name in names:
            for split in splits:
                prepared = prepare_fold(dataset.X, dataset.y, split)
                # Generation is independent of the alert budget, so it is fitted
                # once and reused across the whole operating curve.
                model = _new_model(name, cfg, seed)
                gen_status, gen_error = "ok", ""
                try:
                    with time_limit(cfg.fit_timeout_s), timed() as watch:
                        model.fit_generate(prepared.X_gen, prepared.y_gen)
                except FitTimeout as exc:
                    gen_status, gen_error = "timeout", str(exc)
                except (ValueError, KeyError, RuntimeError, ArithmeticError, MemoryError) as exc:
                    gen_status, gen_error = "failed", f"{type(exc).__name__}: {exc}"

                for alert_rate in cfg.alert_rates:
                    if gen_status != "ok":
                        result = FoldResult(
                            dataset=dataset.name,
                            model=name,
                            seed=seed,
                            fold=split.fold,
                            alert_rate=alert_rate,
                            status=gen_status,
                            error=gen_error,
                        )
                    else:
                        result = run_fold(
                            model,
                            dataset.X,
                            dataset.y,
                            split,
                            cfg,
                            dataset.name,
                            alert_rate,
                            prepared=prepared,
                            generate_seconds=watch.seconds,
                        )
                    if verbose and result.status != "ok":
                        print(
                            f"    [{result.model} fold {split.fold} "
                            f"ar={alert_rate}] {result.error}"
                        )
                    rows.append(result.as_row())
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmarks.run", description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="3 small datasets, minutes")
    mode.add_argument("--full", action="store_true", help="the whole registry")
    parser.add_argument("--datasets", nargs="+", help="explicit dataset names")
    parser.add_argument("--seeds", type=int, help="override the number of seeds")
    parser.add_argument("--outer-folds", type=int, help="override outer fold count")
    parser.add_argument("--max-rows", type=int, help="cap rows per dataset")
    parser.add_argument("--metric", help="selection objective, e.g. f1, f0.5, precision")
    parser.add_argument(
        "--max-candidate-rules",
        type=int,
        help="cap candidates entering combination, equalising pool size across generators",
    )
    parser.add_argument(
        "--selection-split",
        choices=("train", "holdout", "oracle"),
        help="where rules are selected; 'oracle' selects on test and is an upper bound, not an estimate",
    )
    parser.add_argument(
        "--combine-operator", choices=("or", "and"), help="rule composition operator"
    )
    parser.add_argument("--fit-timeout", type=float, help="per-model fit timeout in seconds")
    parser.add_argument(
        "--smallest-first",
        action="store_true",
        help="run smaller (cached-file-size) datasets before larger ones",
    )
    parser.add_argument("--ablations", action="store_true", help="also run ablations")
    parser.add_argument("--verify-registry", action="store_true", help="probe every dataset")
    parser.add_argument("--out", type=Path, help="results directory override")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _smallest_first(names: list[str]) -> list[str]:
    """Order by cached parquet size (a proxy for row count); uncached names run last.

    Useful for a first pass over the whole registry: small/fast datasets surface
    failures quickly, and the slowest (largest) datasets don't block a checkpoint
    dump for everything that ran before them.
    """
    def size(name: str) -> float:
        matches = sorted(CACHE_DIR.glob(f"{name}__v*.parquet"))
        return matches[0].stat().st_size if matches else float("inf")

    return sorted(names, key=size)


def _resolve(args: argparse.Namespace) -> tuple[ExperimentConfig, list[str], str]:
    smoke = args.smoke or not args.full
    cfg = SMOKE_CONFIG if smoke else FULL_CONFIG
    mode = "smoke" if smoke else "full"
    if args.seeds is not None:
        cfg = cfg.with_seed_count(args.seeds)
    replacements: dict[str, Any] = {}
    if args.outer_folds is not None:
        replacements["n_outer_folds"] = args.outer_folds
    if args.max_rows is not None:
        replacements["max_rows"] = args.max_rows
    if args.out is not None:
        replacements["results_dir"] = args.out
    if args.metric is not None:
        replacements["metric"] = args.metric
    if args.max_candidate_rules is not None:
        from dataclasses import replace as _replace

        replacements["selection"] = _replace(
            replacements.get("selection", cfg.selection),
            max_candidate_rules=args.max_candidate_rules,
        )
    if args.selection_split is not None:
        replacements["selection_split"] = args.selection_split
    if args.combine_operator is not None:
        from dataclasses import replace as _replace

        replacements["selection"] = _replace(
            cfg.selection, combine_operator=args.combine_operator
        )
    if args.fit_timeout is not None:
        replacements["fit_timeout_s"] = args.fit_timeout
    if replacements:
        from dataclasses import replace

        cfg = replace(cfg, **replacements)
    names = args.datasets or (list(SMOKE) if smoke else list(REGISTRY))
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}")
    if args.smallest_first:
        names = _smallest_first(names)
    return cfg, names, mode


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    verbose = not args.quiet

    if args.verify_registry:
        report = verify_registry(args.datasets)
        print(report)
        return 0

    cfg, names, mode = _resolve(args)
    run_dir = cfg.results_dir / f"{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(cfg, names, mode)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    availability = availability_report(cfg.baselines)
    write_table(availability, run_dir, "baseline_availability")
    if verbose:
        print(f"Run directory: {run_dir}")
        print(availability)

    write_table(registry_table(), run_dir, "dataset_registry")

    if verbose:
        print(f"Running {mode} benchmark over {len(names)} dataset(s)...")
    results, dataset_meta = run_benchmark(
        cfg, names, verbose=verbose, checkpoint_dir=run_dir
    )
    write_table(dataset_meta, run_dir, "dataset_meta")

    if results.is_empty():
        print("No results produced.", file=sys.stderr)
        return 1

    written = build_report(results, run_dir)

    if args.ablations:
        ablation_rows: list[pl.DataFrame] = []
        for name in names:
            try:
                dataset = load_dataset(name, max_rows=cfg.max_rows, seed=cfg.seeds[0])
            except (DatasetLoadError, OSError, ValueError, KeyError) as exc:
                if verbose:
                    print(f"  [skip ablations] {name}: {exc}")
                continue
            ablation_rows.append(run_ablations(dataset, cfg, cfg.primary_alert_rate))
        frames = [f for f in ablation_rows if not f.is_empty()]
        if frames:
            write_table(pl.concat(frames, how="diagonal_relaxed"), run_dir, "ablations")

    if verbose:
        _print_summary(results, run_dir, written)
    return 0


def _print_summary(
    results: pl.DataFrame, run_dir: Path, written: dict[str, list[Path]]
) -> None:
    ok = results.filter(pl.col("status") == "ok")
    print(f"\nRuns: {results.height} total, {ok.height} succeeded")
    if not ok.is_empty():
        print(
            ok.group_by("model")
            .agg(
                pl.col("test_average_precision").mean().round(4).alias("AP"),
                pl.col("complexity_conditions").mean().round(1).alias("conditions"),
                pl.col("generalisation_gap").mean().round(4).alias("gap"),
                pl.len().alias("n"),
            )
            .sort("AP", descending=True)
        )
    failed = results.filter(pl.col("status") != "ok")
    if not failed.is_empty():
        print("\nFailures by model:")
        print(failed.group_by("model").agg(pl.len().alias("n"), pl.col("error").first()))
    print(f"\nTables written to {run_dir}:")
    for name, paths in written.items():
        if paths:
            print(f"  {name}: {', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    raise SystemExit(main())
