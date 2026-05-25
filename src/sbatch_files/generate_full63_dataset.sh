#!/bin/bash
#SBATCH --job-name=f63_gen_data
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_gen_data_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_gen_data_err_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python full_63_heli_kin_reconstruct/generate_dataset.py \
        --daic \
        --force
