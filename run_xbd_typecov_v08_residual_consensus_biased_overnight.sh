#!/bin/bash
#SBATCH --job-name=xbd-v08-resbias
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=xbd-v08-resbias-%j.out
#SBATCH --error=xbd-v08-resbias-%j.err

set -euo pipefail
cd /data/hlu922/disaster-pfl-pfllib-eval

export TMPDIR=/data/$USER/
export HOME=/data/$USER/
source /data/$USER/miniconda3/etc/profile.d/conda.sh
conda activate pfllib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

DATASET="dataset/xBD_type_coverage_balanced"
ROOT="outputs/xbd_typecov_v08_residual_consensus_biased_${SLURM_JOB_ID}"

echo "DATASET=$DATASET"
echo "ROOT=$ROOT"
echo "START $(date)"

run_v08 () {
  NAME="$1"; shift
  echo
  echo "=== RUN $NAME ==="
  python scripts/xbd/run_ga_dnc_pfl_v08_residual_consensus.py \
    --dataset-dir "$DATASET" \
    --out-dir "$ROOT/$NAME" \
    --rounds 80 \
    --data-order sequential_stream \
    --stream-window-batches 5 \
    --batch-size 32 \
    --device cuda \
    "$@"
}

run_v06 () {
  NAME="$1"; shift
  echo
  echo "=== RUN $NAME ==="
  python scripts/xbd/run_ga_dnc_pfl_v06.py \
    --dataset-dir "$DATASET" \
    --out-dir "$ROOT/$NAME" \
    --rounds 80 \
    --data-order sequential_stream \
    --stream-window-batches 5 \
    --batch-size 32 \
    --device cuda \
    "$@"
}

# Baselines
run_v06 local_80r --mode local
run_v06 residual_local_80r --mode residual_local
run_v06 ga_dnc_fedavg_top2_tau0_lam01_80r --mode ga_dnc --global-agg fedavg --neighbour-mode dynamic --top-k 2 --tau 0.0 --assist-lambda 0.1

# Residual consensus without global bias
run_v08 v08_rescon_top1_tau03_lam005_beta0 --mode ga_dnc_residual_consensus --neighbour-mode dynamic --top-k 1 --tau 0.3 --assist-lambda 0.05 --gate-margin 0.0 --global-bias-beta 0.0
run_v08 v08_rescon_top2_tau03_lam005_beta0 --mode ga_dnc_residual_consensus --neighbour-mode dynamic --top-k 2 --tau 0.3 --assist-lambda 0.05 --gate-margin 0.0 --global-bias-beta 0.0

# Soft collaboration-biased global anchor: all clients non-zero, mild incoming-neighbour boost
run_v08 v08_rescon_top1_tau03_lam005_beta01 --mode ga_dnc_residual_consensus --neighbour-mode dynamic --top-k 1 --tau 0.3 --assist-lambda 0.05 --gate-margin 0.0 --global-bias-beta 0.1
run_v08 v08_rescon_top1_tau03_lam005_beta02 --mode ga_dnc_residual_consensus --neighbour-mode dynamic --top-k 1 --tau 0.3 --assist-lambda 0.05 --gate-margin 0.0 --global-bias-beta 0.2
run_v08 v08_rescon_top1_tau03_lam005_beta03 --mode ga_dnc_residual_consensus --neighbour-mode dynamic --top-k 1 --tau 0.3 --assist-lambda 0.05 --gate-margin 0.0 --global-bias-beta 0.3
run_v08 v08_rescon_top2_tau03_lam005_beta02 --mode ga_dnc_residual_consensus --neighbour-mode dynamic --top-k 2 --tau 0.3 --assist-lambda 0.05 --gate-margin 0.0 --global-bias-beta 0.2
run_v08 v08_rescon_top1_tau03_lam01_beta02 --mode ga_dnc_residual_consensus --neighbour-mode dynamic --top-k 1 --tau 0.3 --assist-lambda 0.1 --gate-margin 0.0 --global-bias-beta 0.2

# Random control with same bias rule
run_v08 v08_rescon_random_top1_tau03_lam005_beta02 --mode ga_dnc_residual_consensus --neighbour-mode random --top-k 1 --tau 0.3 --assist-lambda 0.05 --gate-margin 0.0 --global-bias-beta 0.2

python - <<'PY'
import csv, os
from pathlib import Path
root = Path("outputs") / f"xbd_typecov_v08_residual_consensus_biased_{os.environ['SLURM_JOB_ID']}"
rows=[]
for d in sorted(root.iterdir()):
    p=d/"round_metrics.csv"
    if p.exists():
        rr=list(csv.DictReader(open(p)))
        if rr:
            r=rr[-1]; r["run_name"]=d.name; rows.append(r)
fields=["run_name"]; seen=set(fields)
for r in rows:
    for k in r:
        if k not in seen:
            seen.add(k); fields.append(k)
out=root/"summary_final.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print("summary:", out)
for r in sorted(rows,key=lambda x:float(x.get("avg_accuracy",0)),reverse=True):
    print(r["run_name"], "acc", r.get("avg_accuracy"), "macro_f1", r.get("avg_macro_f1"), "worst", r.get("worst_client_accuracy"), "damaged", r.get("avg_damaged_recall"), "accepted", r.get("accepted_assist_clients",""), "rejected", r.get("rejected_assist_clients",""))
PY

python scripts/xbd/analyze_ga_dnc_collaboration.py --run-root "$ROOT" || true

echo "END $(date)"
