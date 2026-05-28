#!/bin/bash
#SBATCH --job-name=f63_synth_s1-50_s2-300
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_synth_s1-50_s2-300_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/f63_synth_s1-50_s2-300_err_%j.log
#SBATCH --time=02:00:00
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
        --stage1-epochs 50 \
        --stage2-epochs 300
