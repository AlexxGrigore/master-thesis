"""Merge the creation + training profiles into one slide-ready table (CSV + stdout).

Usage:
    python profiling_experiment/summarize.py \
        --creation outputs/.../creation_<ts>.json \
        --training outputs/.../training_<ts>.json

If paths are omitted, picks the most recent creation_*.json / training_*.json under
outputs/new_mapping_function/profiling_experiment/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import paths  # noqa: E402  (sibling)


def _latest(folder: pathlib.Path, prefix: str) -> pathlib.Path | None:
    files = sorted(folder.glob(f"{prefix}_*.json"))
    return files[-1] if files else None


def _load(path: pathlib.Path | None) -> dict:
    if path is None or not path.exists():
        return {"records": {}}
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize profiling results into a CSV.")
    p.add_argument("--creation", type=pathlib.Path, default=None)
    p.add_argument("--training", type=pathlib.Path, default=None)
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--daic", action="store_true", help="(accepted for symmetry; output dir is repo-relative)")
    args = p.parse_args()

    folder = paths.output_dir()

    creation = _load(args.creation or _latest(folder, "creation"))
    training = _load(args.training or _latest(folder, "training"))
    out_csv = args.out or (folder / "summary.csv")

    sizes = sorted({
        rec["n_heliostats"]
        for src in (creation, training)
        for rec in src["records"].values()
        if "n_heliostats" in rec
    })

    rows = []
    for n in sizes:
        c = creation["records"].get(f"create_N{n}", {})
        t = training["records"].get(f"train_N{n}", {})
        rows.append({
            "n_heliostats": n,
            "create_seconds": c.get("seconds"),
            "create_peak_vram_gb": c.get("peak_vram_alloc_gb"),
            "train_seconds": t.get("seconds"),
            "train_raytrace_seconds": t.get("raytrace_seconds"),
            "train_raytrace_pct": t.get("raytrace_share_pct"),
            "train_peak_vram_gb": t.get("peak_vram_alloc_gb"),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    gpu = training.get("gpu_name") or creation.get("gpu_name") or "?"
    print(f"GPU: {gpu}")
    header = ("  N | create(s) | create VRAM | train(s) | raytrace(s) | rt% | train VRAM")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['n_heliostats']:>3} | "
              f"{str(r['create_seconds']):>9} | "
              f"{str(r['create_peak_vram_gb']):>11} | "
              f"{str(r['train_seconds']):>8} | "
              f"{str(r['train_raytrace_seconds']):>11} | "
              f"{str(r['train_raytrace_pct']):>3} | "
              f"{str(r['train_peak_vram_gb'])}")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
