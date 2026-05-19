#!/bin/bash
#SBATCH --job-name=f63_real_pixel
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_real_pixel_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_real_pixel_err_%j.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python full_63_heli_kin_reconstruct/main.py \
        --daic \
        --dataset-type real \
        --loss-type pixel \
        --stage1-epochs 100 \
        --stage2-epochs 500
