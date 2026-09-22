import sys, argparse
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.config import SMOKE_CONFIG
from benchmarks.run import run_benchmark

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--out', required=True); ap.add_argument('--drop', nargs='*', default=[])
ap.add_argument('--only', help='run a single model (e.g. figs, iguanas)')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
suffix = f"-{a.only}" if a.only else ""
target = out / f"{a.dataset}-seed{a.seed}{suffix}.csv"
if target.exists(): print(f"skip {target.name}"); sys.exit(0)
base = tuple(b for b in SMOKE_CONFIG.baselines if b not in a.drop)
cfg = replace(SMOKE_CONFIG, seeds=(a.seed,), baselines=base)
if a.only:
    import benchmarks.run as _run
    cfg = replace(cfg, baselines=tuple(b for b in base if b == a.only))
    _run.live_model_names = lambda c, _m=a.only: [_m]
rows, _ = run_benchmark(cfg, [a.dataset], verbose=False)
if rows.height: rows.write_csv(target); print(f"done {target.name}: {rows.height}")
else: print(f"EMPTY {a.dataset}-{a.seed}")
