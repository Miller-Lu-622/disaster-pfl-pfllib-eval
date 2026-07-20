#!/bin/bash
#SBATCH --job-name=pfllib-xbd9-fix-failed
#SBATCH --output=pfllib-xbd9-fix-failed-%j.out
#SBATCH --error=pfllib-xbd9-fix-failed-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -uo pipefail

export TMPDIR=/data/$USER/
export HOME=/data/$USER/
source /data/$USER/miniconda3/etc/profile.d/conda.sh
conda activate pfllib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/$USER/disaster-pfl-pfllib-eval/system

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start: $(date)"
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('CUDA device 0:', torch.cuda.get_device_name(0))
PY

python - <<'PY'
from pathlib import Path
from datetime import datetime
root = Path.cwd()
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def patch_file(rel, replacements):
    p = root / rel
    text = p.read_text()
    orig = text
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
            print(f"[PATCH] {rel}")
        elif new in text:
            print(f"[ALREADY PATCHED] {rel}")
        else:
            print(f"[WARN] pattern not found: {rel}")
    if text != orig:
        bak = p.with_suffix(p.suffix + f'.bak_fix_failed_{stamp}')
        bak.write_text(orig)
        p.write_text(text)
        print(f"[OK] backup={bak}")

binary_auc_new = """if self.num_classes == 2:
            y_true_1d = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
            y_score = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            auc = metrics.roc_auc_score(y_true_1d, y_score)
        else:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')"""

patch_file('flcore/clients/clientapfl.py', [
    ("auc = metrics.roc_auc_score(y_true, y_prob, average='micro')", binary_auc_new)
])
patch_file('flcore/clients/clientditto.py', [
    ("auc = metrics.roc_auc_score(y_true, y_prob, average='micro')", binary_auc_new)
])
patch_file('flcore/servers/serveras.py', [
    ("""print(f'+++++++++++++++++++++++++++++++++++++++++')
        gen_acc = self.avg_generalization_metrics()
        print(f'Generalization Acc: {gen_acc}')
        print(f'+++++++++++++++++++++++++++++++++++++++++')""",
     """print(f'+++++++++++++++++++++++++++++++++++++++++')
        if hasattr(self, 'avg_generalization_metrics'):
            gen_acc = self.avg_generalization_metrics()
            print(f'Generalization Acc: {gen_acc}')
        else:
            print('Generalization metrics unavailable; skipping.')
        print(f'+++++++++++++++++++++++++++++++++++++++++')""")
])
PY

python -m py_compile flcore/clients/clientapfl.py flcore/clients/clientditto.py flcore/servers/serveras.py || exit 1

DATASET=xBD_event_balanced
MODEL=CNN
ROUNDS=50
LOCAL_STEPS=1
NUM_CLIENTS=9
NUM_CLASSES=2
JOIN_RATIO=1.0
DEVICE=cuda
GPU_ID=0
LOGDIR=../logs/xBD_event_balanced_9clients_binary2_FIX_FAILED_50r_${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

COMMON=(-data "$DATASET" -m "$MODEL" -gr "$ROUNDS" -ls "$LOCAL_STEPS" -nc "$NUM_CLIENTS" -ncl "$NUM_CLASSES" -jr "$JOIN_RATIO" -t 1 -did "$GPU_ID" -dev "$DEVICE")

echo "algorithm,status,exit_code,final_accuracy,final_auc,best_accuracy,log_file,note" > "$LOGDIR/fix_status.csv"

extract_metrics() {
  local LOG="$1"
  python - "$LOG" <<'PY'
import sys,re
text=open(sys.argv[1],errors='ignore').read().splitlines()
acc=[]; auc=[]
for l in text:
    m=re.search(r'Averaged? Test Accuracy:\s*([0-9.\-eE]+)',l)
    if m: acc.append(float(m.group(1)))
    m=re.search(r'Averaged? Test AUC:\s*([0-9.\-eE]+)',l)
    if m: auc.append(float(m.group(1)))
fa=acc[-1] if acc else ''
fu=auc[-1] if auc else ''
ba=max(acc) if acc else ''
print(f"{fa},{fu},{ba}")
PY
}

run_one() {
  ALG="$1"; shift
  LOG="$LOGDIR/${ALG}_fix_50r.log"
  echo "=== Running $ALG at $(date) ==="
  timeout --foreground 60m python -u main.py "${COMMON[@]}" -algo "$ALG" "$@" > "$LOG" 2>&1
  code=$?
  if [ $code -eq 0 ]; then status="OK"; else status="FAILED"; fi
  metrics=$(extract_metrics "$LOG")
  echo "$ALG,$status,$code,$metrics,$LOG,$*" >> "$LOGDIR/fix_status.csv"
  echo "=== Finished $ALG status=$status code=$code metrics=$metrics at $(date) ==="
}

run_one APFL
run_one Ditto
run_one FedAS
run_one FedCAC -bt 1
run_one FD -lr 0.001
run_one pFedMe -lr 0.001 -ls 5
run_one FedGC

cd /data/$USER/disaster-pfl-pfllib-eval
PKG="pfllib_xbd_eventbalanced_binary2_FIX_FAILED_${SLURM_JOB_ID}_logs_results.tar.gz"
tar -czf "$PKG" \
  "logs/xBD_event_balanced_9clients_binary2_FIX_FAILED_50r_${SLURM_JOB_ID}" \
  results/*xBD_event_balanced* 2>/dev/null || true

echo "Packaged artifact:"
ls -lh "$PKG" || true
echo "Final fix status:"
cat "logs/xBD_event_balanced_9clients_binary2_FIX_FAILED_50r_${SLURM_JOB_ID}/fix_status.csv"
echo "End: $(date)"
