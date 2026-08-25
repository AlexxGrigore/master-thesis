#!/bin/bash
#SBATCH --job-name=fel_daic_matrix
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/fel_daic_matrix_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/fel_daic_matrix_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

# FEL encoder-only experiment matrix on DAIC (synthetic balanced_dataset, 62
# heliostats). Levers under test, all on top of the stage-1 warm start:
#   1. fel_daic_long      — 300 epochs, defaults          (longer schedule)
#   2. fel_daic_gain05    — 300 epochs, OUTPUT_GAIN=0.05  (larger Δθ steps)
#   3. fel_daic_pix30     — 300 epochs, pixel loss λ≈0.3  (stronger aux loss)
# Each run is followed by its test-split evaluation. Adjust EPOCHS or drop
# lines as needed; each 300-epoch run is expected to take ~1-3 h on an A40.

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs
cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

SIF=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif
CKPT_DIR=../outputs/fine_error_learning/all63_stage1_synth

# Step 0 (once, ~6 min): synthetic stage-1 checkpoints, skipped if present.
if [ ! -f "$CKPT_DIR/AB43/stage1_checkpoint.pt" ]; then
    apptainer exec --nv --bind /tudelft.net:/tudelft.net $SIF \
        python one_heliostat_demo/run_all.py --daic \
            --data-mode synthetic --skip-dataset-gen --skip-stage2 --no-plots \
            --output-dir "$CKPT_DIR"
fi

for SPEC in \
    "fel_daic_long::" \
    "fel_daic_gain05:--output-gain 0.05:" \
    "fel_daic_pix30:--pixel-loss 0.3:"
do
    NAME="${SPEC%%:*}"; REST="${SPEC#*:}"; EXTRA="${REST%%:*}"
    apptainer exec --nv --bind /tudelft.net:/tudelft.net $SIF \
        python fine_error_learning/main.py --daic \
            --run-name "$NAME" --epochs 300 $EXTRA
    apptainer exec --nv --bind /tudelft.net:/tudelft.net $SIF \
        python fine_error_learning/main.py --daic \
            --evaluate "../outputs/fine_error_learning/$NAME" --eval-split test
done
