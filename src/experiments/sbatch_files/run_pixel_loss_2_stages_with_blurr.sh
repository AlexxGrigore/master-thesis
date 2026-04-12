#!/bin/bash
#SBATCH --job-name=pixel_blurr
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/pixel_blurr_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/pixel_blurr_err_%j.log
#SBATCH --time=2:12:33
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1


mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python experiments/kr_train_pixel_loss_blurr/main.py