#!/bin/bash
#SBATCH --job-name=no_defl_focal_spot
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/no_defl_focal_spot_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/no_defl_focal_spot_err_%j.log
#SBATCH --time=1:31:36
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python no_deflectometry/kr_training_focal_spot/main.py
