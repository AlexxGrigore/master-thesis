#!/bin/bash
#SBATCH --job-name=no_defl_alignment
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/no_defl_alignment_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/no_defl_alignment_err_%j.log
#SBATCH --time=1:00:01
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python no_deflectometry/kr_train_alignment_loss/main.py
