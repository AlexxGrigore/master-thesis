#!/bin/bash
#SBATCH --job-name=train_sizes_balanced
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/train_sizes_balanced_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/train_sizes_balanced_err_%j.log
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python one_heliostat_demo/run_all_train_sizes.py \
        --split-type balanced \
        --all-heliostats \
        --daic
