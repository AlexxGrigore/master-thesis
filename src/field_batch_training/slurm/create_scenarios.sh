#!/bin/bash
#SBATCH --job-name=field_scenarios
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/field_scen_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/field_scen_err_%j.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Build the 13 batched field scenarios (field_batch_00..12.h5, alphabetical,
# <=100 heliostats each, ideal surfaces) from the downloaded PAINT data.
# Run AFTER download_dataset.sh has finished:
#
#   sbatch field_batch_training/slurm/create_scenarios.sh
#
# Idempotent: existing batch files are skipped (use --force inside to rebuild).
# Output: /home/nfs/agrigore/.../master-thesis/scenarios/field_batches/

set -e

REPO=/home/nfs/agrigore/projects/githubProjects/master-thesis
SIF=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif

mkdir -p "$REPO/logs"
cd "$REPO/src"

apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" \
    python field_batch_training/build_batch_scenarios.py --daic

echo "=== verification ==="
apptainer exec --bind /tudelft.net:/tudelft.net "$SIF" python - <<EOF
import h5py, pathlib
d = pathlib.Path("$REPO/scenarios/field_batches")
files = sorted(d.glob("field_batch_*.h5"))
total = 0
for f in files:
    with h5py.File(f, "r") as h:
        n = len(h["heliostats"].keys())
    total += n
    print(f"{f.name}: {n} heliostats")
print(f"batches: {len(files)} (expect 13), heliostats: {total} (expect 1277)")
EOF
