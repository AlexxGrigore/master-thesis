#!/bin/bash
#SBATCH --job-name=ftp_real_poly3
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/ftp_real_poly3_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/ftp_real_poly3_err_%j.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python full_training_pipeline/main.py --dataset-type real --model-type poly3 --daic
