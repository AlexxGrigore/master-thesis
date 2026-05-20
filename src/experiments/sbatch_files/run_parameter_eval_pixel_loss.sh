#!/bin/bash
#SBATCH --job-name=param_eval_pixel
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/param_eval_pixel_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/param_eval_pixel_err_%j.log
#SBATCH --time=11:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python experiments/parameter_evaluation_pixel_loss/main.py
