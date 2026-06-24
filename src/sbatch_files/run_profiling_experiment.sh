#!/bin/bash
#SBATCH --job-name=artist_profiling
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/profiling_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/profiling_err_%j.log
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1

# ARTIST upscaling cost experiment.
#   Phase A: build scenarios for N = 1, 10, 20, 50 heliostats (NURBS fit)  -> time + VRAM
#   Phase B: joint kinematics training on each scenario                    -> time + VRAM
#   Then merge both into a slide-ready CSV.
# All three steps run sequentially in one job so the scenarios built in A feed B.

set -e

REPO=/home/nfs/agrigore/projects/githubProjects/master-thesis
SIF=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif
SIZES="1 10 20 50"

mkdir -p "$REPO/logs"
cd "$REPO/src"

run() {
    apptainer exec --nv --bind /tudelft.net:/tudelft.net "$SIF" python "$@"
}

echo "=== Phase A: scenario creation ==="
run profiling_experiment/create_scenarios.py --daic --sizes $SIZES

echo "=== Phase B: joint training ==="
run profiling_experiment/run_training.py --daic --sizes $SIZES

echo "=== Summary ==="
run profiling_experiment/summarize.py --daic

# Record which GPU actually ran this, for the slides.
nvidia-smi --query-gpu=name,memory.total --format=csv
