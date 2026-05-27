#!/bin/bash
# Array job: one task per seed (10 seeds × 5 heliostats × 8 train sizes).
# Each task runs independently in parallel on its own GPU.
#
# Submit:
#   sbatch sbatch_files/run_one_hel_multi_seed.sh
#
# For the ideal-scenario variant:
#   sbatch sbatch_files/run_one_hel_multi_seed.sh --no-deflectometry
#
# NOTE: 1h may be tight depending on GPU speed. Increase --time if jobs time out.

#SBATCH --job-name=one_hel_mseed
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/one_hel_mseed_%A_%a.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/one_hel_mseed_%A_%a.log
#SBATCH --array=0-9
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python one_hel_multi_seed/main.py \
        --daic \
        --seed-index "$SLURM_ARRAY_TASK_ID" \
        "$@"
