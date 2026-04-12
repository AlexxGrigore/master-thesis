#!/bin/bash
#SBATCH --job-name=focal_spot
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/focal_spot_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/focal_spot_err_%j.log
#SBATCH --time=1:32:30
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python experiments/kr_training_focal_spot/main.py