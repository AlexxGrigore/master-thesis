#!/bin/bash
#SBATCH --job-name=kinematic_training_no_defl
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/output_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/error_%j.log
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python no_deflectometry/kr_training_focal_spot/main.py
