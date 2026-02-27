#!/bin/bash
#SBATCH --job-name=kinematic_training
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/src/logs/output_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/src/logs/error_%j.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/src/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python kinematic_reconstruction_training/main.py