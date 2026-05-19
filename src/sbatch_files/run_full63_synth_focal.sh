#!/bin/bash
#SBATCH --job-name=f63_synth_focal
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_synth_focal_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_synth_focal_err_%j.log
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec --nv \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python full_63_heli_kin_reconstruct/main.py \
        --daic \
        --dataset-type synthetic \
        --loss-type focal_spot \
        --stage1-epochs 100 \
        --stage2-epochs 300
