#!/bin/bash
#SBATCH --job-name=gen_data_azimuth
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/gen_data_azimuth_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/gen_data_azimuth_err_%j.log
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python one_heliostat_demo/generate_all.py \
        --split-type azimuth \
        --daic
