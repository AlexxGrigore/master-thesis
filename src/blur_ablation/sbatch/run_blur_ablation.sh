#!/bin/bash
#SBATCH --job-name=blur_ablation_sweep
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/blur_ablation_sweep_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/blur_ablation_sweep_err_%j.log
#SBATCH --time=0:01:24
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python blur_ablation/main.py
