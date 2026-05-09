#!/bin/bash
#SBATCH --job-name=ff200_real
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/ff200_real_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/ff200_real_err_%j.log
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python full_field_200_samples/main.py --dataset-type real --daic
