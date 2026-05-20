#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/check_env_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/check_env_err_%j.log
#SBATCH --time=0:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python check_env.py
