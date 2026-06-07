import json
import re
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python scripts/parse_pfllib_log_generic.py <log_path> <output_json>")
    sys.exit(1)

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
text = log_path.read_text(errors="ignore")

result = {
    "log_file": str(log_path),
    "algorithm": None,
    "dataset": None,
    "model": None,
    "rounds": [],
    "best_accuracy": None,
    "avg_time_per_round_sec": None,
    "total_time_sec": None,
    "result_file": None,
    "final_memory": {}
}

for key, field in [("algorithm", "algorithm"), ("dataset", "dataset"), ("model", "model")]:
    m = re.search(rf"^{key} = (.+)$", text, re.M)
    if m:
        result[field] = m.group(1).strip()

round_pattern = re.compile(
    r"-------------Round number: (\d+)-------------.*?"
    r"Averaged Train Loss: ([0-9.]+).*?"
    r"Averaged Test Accuracy: ([0-9.]+).*?"
    r"Averaged Test AUC: ([0-9.]+).*?"
    r"Std Test Accuracy: ([0-9.]+).*?"
    r"Std Test AUC: ([0-9.]+).*?"
    r"time cost ------------------------- ([0-9.]+)",
    re.S
)

for m in round_pattern.finditer(text):
    result["rounds"].append({
        "round": int(m.group(1)),
        "avg_train_loss": float(m.group(2)),
        "avg_test_accuracy": float(m.group(3)),
        "avg_test_auc": float(m.group(4)),
        "std_test_accuracy": float(m.group(5)),
        "std_test_auc": float(m.group(6)),
        "time_cost_sec": float(m.group(7)),
    })

m = re.search(r"Best accuracy\.\s*([0-9.]+)", text)
if m:
    result["best_accuracy"] = float(m.group(1))

m = re.search(r"Average time cost per round\.\s*([0-9.]+)", text)
if m:
    result["avg_time_per_round_sec"] = float(m.group(1))

m = re.search(r"Average time cost: ([0-9.]+)s\.", text)
if m:
    result["total_time_sec"] = float(m.group(1))

m = re.search(r"File path: (.+)", text)
if m:
    result["result_file"] = m.group(1).strip()

m = re.search(r"Total Tensors: (\d+) Used Memory: ([0-9.]+)M", text)
if m:
    result["final_memory"]["total_tensors"] = int(m.group(1))
    result["final_memory"]["used_memory_mb"] = float(m.group(2))

m = re.search(r"The allocated memory on cuda:0: ([0-9.]+)M", text)
if m:
    result["final_memory"]["allocated_cuda0_mb"] = float(m.group(1))

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=2))
print(f"Wrote {out_path}")
