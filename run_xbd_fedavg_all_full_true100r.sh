#!/bin/bash
#SBATCH --job-name=xbd-all100t
#SBATCH --output=logs/xbd-all100t-%j.out
#SBATCH --error=logs/xbd-all100t-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

export TMPDIR=/data/$USER/
export HOME=/data/$USER/
source /data/$USER/miniconda3/etc/profile.d/conda.sh
conda activate pfllib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/hlu922/disaster-pfl-pfllib-eval
mkdir -p logs results/xbd_all_full_true100r

python scripts/xbd/train_xbd_fedavg_all_full_100r_record.py
