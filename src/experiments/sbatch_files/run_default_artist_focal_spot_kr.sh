#!/bin/bash
#SBATCH --job-name=artist_baseline
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/artist_baseline_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/artist_baseline_err_%j.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python experiments/default_artist_focal_spot_kr/main.py
