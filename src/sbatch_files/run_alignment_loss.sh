#!/bin/bash
#SBATCH --job-name=alignment
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/alignment_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/alignment_err_%j.log
#SBATCH --time=1:02:17
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python kr_train_alignment_loss/main.py