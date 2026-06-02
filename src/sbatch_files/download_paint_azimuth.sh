#!/bin/bash
#SBATCH --job-name=paint_download
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/paint_download_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/paint_download_err_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Note: this job downloads data from the internet.
# If DAIC compute nodes do not have outbound internet access, run this
# directly on the login node instead:
#   cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src
#   apptainer exec --bind /tudelft.net:/tudelft.net \
#       /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
#       python download_paint_benchmark_200.py --split-type azimuth --daic

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src

apptainer exec \
    --bind /tudelft.net:/tudelft.net \
    /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
    python download_paint_benchmark_200.py \
        --split-type azimuth \
        --daic
