#!/bin/bash
#SBATCH --job-name=improved_focal_spot_kr
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/improved_focal_spot_kr_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/improved_focal_spot_kr_err_%j.log
#SBATCH --time=4:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python experiments/improved_focal_spot_kr/main.py
