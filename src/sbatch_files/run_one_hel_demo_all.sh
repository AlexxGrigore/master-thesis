#!/bin/bash
#SBATCH --job-name=one_hel_demo_all
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/one_hel_demo_all_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/one_hel_demo_all_err_%j.log
#SBATCH --time=02:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python one_heliostat_demo/run_all.py \
        --skip-dataset-gen \
        --daic
