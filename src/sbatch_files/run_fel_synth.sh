#!/bin/bash
#SBATCH --job-name=fel_synth
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/fel_synth_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/fel_synth_err_%j.log
#SBATCH --time=08:00:00
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

# Fine Error Learning on the synthetic balanced_dataset.
# Step 1 (fast, ~6 min): synthetic stage-1 checkpoints, skipped if present.
# Step 2: FEL training (shared transformer, end-to-end through ARTIST).

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

SIF=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif
CKPT_DIR=../outputs/fine_error_learning/all63_stage1_synth

if [ ! -f "$CKPT_DIR/AB43/stage1_checkpoint.pt" ]; then
    apptainer exec --nv --bind /tudelft.net:/tudelft.net $SIF \
        python one_heliostat_demo/run_all.py \
            --data-mode synthetic --skip-dataset-gen --skip-stage2 --no-plots \
            --output-dir "$CKPT_DIR"
fi

apptainer exec --nv --bind /tudelft.net:/tudelft.net $SIF \
    python fine_error_learning/main.py --run-name fel_synth_full

# Evaluation on the held-out test split (before/after Δθ, parameter recovery).
apptainer exec --nv --bind /tudelft.net:/tudelft.net $SIF \
    python fine_error_learning/main.py \
        --evaluate ../outputs/fine_error_learning/fel_synth_full --eval-split test
