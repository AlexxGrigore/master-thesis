#!/bin/bash
#SBATCH --job-name=build_artist_sif
#SBATCH --output=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/build_sif_out_%j.log
#SBATCH --error=/home/nfs/agrigore/projects/githubProjects/master-thesis/logs/build_sif_err_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=64G
# NOTE: building a container needs NO GPU (the --nv flag is only for *running* it).
# This is a CPU/disk/network job: it pulls python:3.10 and pip-installs torch+cu124.

set -e

BUILD_DIR=/tudelft.net/staff-umbrella/StudentsCVlab/agrigore
DEF=artist-local.def
SIF=artist-local.sif

mkdir -p /home/nfs/agrigore/projects/githubProjects/master-thesis/logs

# --- scratch space -----------------------------------------------------------
# The build extracts the base image and installs several GB of packages, so it
# needs real scratch space. Prefer fast node-local scratch (requested via --tmp
# above). If you hit "no space left on device", either raise --tmp, or point these
# two vars at a folder on the umbrella share instead:
#   export APPTAINER_TMPDIR=$BUILD_DIR/.apptainer_tmp
#   export APPTAINER_CACHEDIR=$BUILD_DIR/.apptainer_cache
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer_build_${SLURM_JOB_ID}"
export APPTAINER_CACHEDIR="${TMPDIR:-/tmp}/apptainer_cache_${SLURM_JOB_ID}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
trap 'rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"' EXIT

cd "$BUILD_DIR"

echo "Host        : $(hostname)"
echo "Build dir   : $(pwd)"
echo "Apptainer   : $(command -v apptainer) $(apptainer --version 2>/dev/null)"
echo "Definition  : $DEF  ->  $SIF"
echo "Scratch     : $APPTAINER_TMPDIR"
echo

# --force overwrites the existing artist-local.sif. The def's %files section copies
# host paths (/home/nfs/agrigore/projects/ARTIST and .../PAINT) into the image, so
# this must run on a node that can see your home directory (compute nodes can).
#
# If the build fails with a privilege / "fakeroot" error, retry with:
#     apptainer build --fakeroot --force "$SIF" "$DEF"
apptainer build --force "$SIF" "$DEF"

echo
echo "Done. Image at $BUILD_DIR/$SIF"
ls -lh "$BUILD_DIR/$SIF"
