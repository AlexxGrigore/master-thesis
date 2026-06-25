#!/bin/bash
#SBATCH --job-name=raytracer_bench
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/raytracer_bench_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/raytracer_bench_err_%j.log
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

# Ray-tracer-only cost benchmark: peak VRAM + time of a single ray-tracing pass at
# 25x25 vs 50x50 surface points, across a few heliostat counts, forward and forward+
# backward. No training, no scenario creation -> runs in a few minutes.

set -e

REPO=/home/nfs/agrigore/projects/githubProjects/master-thesis
SIF=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif

mkdir -p "$REPO/logs"
cd "$REPO/src"

apptainer exec --nv --bind /tudelft.net:/tudelft.net "$SIF" \
    python profiling_experiment/raytracer_benchmark.py \
        --daic \
        --resolutions 25 50 \
        --sizes 1 10 30 63

nvidia-smi --query-gpu=name,memory.total --format=csv
