#!/bin/bash
#SBATCH --job-name=xbd-smoke
#SBATCH --output=logs/xbd-smoke-%j.out
#SBATCH --error=logs/xbd-smoke-%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

export TMPDIR=/data/$USER/
export HOME=/data/$USER/
export XDG_CACHE_HOME=/data/$USER/.cache
export PIP_CACHE_DIR=/data/$USER/.cache/pip
mkdir -p /data/$USER/.cache/pip

export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
export PYTHONUNBUFFERED=1

source /data/$USER/miniconda3/etc/profile.d/conda.sh
conda activate pfllib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/$USER/disaster-pfl-pfllib-eval

python scripts/xbd/train_xbd_smoke.py
