import json
from pathlib import Path

files = [
    "results/fedavg_summary.json",
    "results/fedprox_summary.json",
    "results/fedbn_summary.json",
    "results/ditto_summary.json",
    "results/pfedme_summary.json",
]

rows = []
for f in files:
    p = Path(f)
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    last = data["rounds"][-1] if data["rounds"] else {}
    rows.append({
        "algorithm": data.get("algorithm"),
        "dataset": data.get("dataset"),
        "model": data.get("model"),
        "best_accuracy": data.get("best_accuracy"),
        "final_test_accuracy": last.get("avg_test_accuracy"),
        "final_test_auc": last.get("avg_test_auc"),
        "final_train_loss": last.get("avg_train_loss"),
        "avg_time_per_round_sec": data.get("avg_time_per_round_sec"),
        "total_time_sec": data.get("total_time_sec"),
    })

out_json = Path("results/all_runs_summary.json")
out_json.write_text(json.dumps(rows, indent=2))

out_csv = Path("results/all_runs_summary.csv")
with out_csv.open("w") as f:
    f.write("algorithm,dataset,model,best_accuracy,final_test_accuracy,final_test_auc,final_train_loss,avg_time_per_round_sec,total_time_sec\n")
    for r in rows:
        f.write(
            f"{r['algorithm']},{r['dataset']},{r['model']},{r['best_accuracy']},{r['final_test_accuracy']},{r['final_test_auc']},{r['final_train_loss']},{r['avg_time_per_round_sec']},{r['total_time_sec']}\n"
        )

print(f"Wrote {out_json}")
print(f"Wrote {out_csv}")
