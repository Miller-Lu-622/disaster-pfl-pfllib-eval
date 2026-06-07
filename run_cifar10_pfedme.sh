#!/bin/bash
#SBATCH --job-name=c10-pfedme
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=c10-pfedme.out
#SBATCH --error=c10-pfedme.err
#SBATCH --gres=gpu:1

export TMPDIR=/data/$USER/
export HOME=/data/$USER/
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
export PYTHONUNBUFFERED=1

source /data/$USER/miniconda3/etc/profile.d/conda.sh
conda activate pfllib

cd /data/$USER/disaster-pfl-pfllib-eval/system

python -u main.py \
  -data Cifar10 \
  -m CNN \
  -algo pFedMe \
  -gr 20 \
  -ls 1 \
  -nc 20 \
  -jr 1.0 \
  -t 1 \
  -did 0 \
  -dev cuda \
  -bt 1.0 \
  -lam 15 \
  -K 5 \
  -lrp 0.01
