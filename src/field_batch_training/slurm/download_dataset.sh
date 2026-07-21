#!/bin/bash
#SBATCH --job-name=paint_dl_50_20
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/paint_dl_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/paint_dl_err_%j.log
#SBATCH --time=10:00:00
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Download the 50-20-20 PAINT benchmark (all 1277 heliostats) onto umbrella storage.
#   - calibration_properties + flux_image for train/validation/test (~230k small files;
#     flux kept for a possible future image-based loss — add --skip-flux to halve it)
#   - tower measurements + per-heliostat Properties (needed for scenario creation)
#   - NO deflectometry h5 files (only 63 heliostats have them; large; field trains ideal)
# Idempotent: safe to resubmit after a timeout — finished parts are skipped.
#
# NOTE: needs outbound internet. If downloads fail on a compute node, run the same
# python command directly on the login node inside tmux instead.

set -e

REPO=/home/nfs/agrigore/projects/githubProjects/master-thesis
SIF=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif
PAINT_DIR=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint

mkdir -p "$REPO/logs" "$PAINT_DIR"
cd "$REPO/src"

apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" \
    python download_paint_benchmark.py \
        --split-type balanced \
        --train-size 50 \
        --val-size 20 \
        --paint-dir "$PAINT_DIR" \
        --skip-deflectometry

echo "=== verification ==="
apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" python - <<EOF
import pathlib
P = pathlib.Path("$PAINT_DIR")
b = "benchmark_split-balanced_train-50_validation-20"
csv = P / "splits" / f"{b}.csv"
print("split CSV:", csv.exists(), csv)
for split in ("train", "validation", "test"):
    n = len(list((P / b / "calibration_properties" / split).glob("*.json")))
    m = len(list((P / b / "flux_image" / split).glob("*.png")))
    print(f"{split}: {n} jsons, {m} pngs")
props = len(list((P / "heliostats").glob("*/Properties/*-heliostat-properties.json")))
print("heliostat Properties:", props, "(expect 1277)")
EOF
# Expected: train 63850 / validation 25540 / test 25540 jsons (844 pngs missing upstream is normal).
