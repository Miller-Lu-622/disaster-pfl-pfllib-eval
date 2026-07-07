import pandas as pd

client = pd.read_csv("results/xbd_realistic_50r/client_metrics.csv")
rounds = pd.read_csv("results/xbd_realistic_50r/round_metrics.csv")
summary = pd.read_csv("results/xbd_realistic_50r/selected_client_summary.csv")

latest_round = client["round"].max()
last = client[client["round"] == latest_round].copy()

print("Latest round:", latest_round)

print("\nGlobal metrics latest:")
print(rounds[rounds["round"] == latest_round])

print("\nPer-client latest sorted by damage_ratio:")
print(last.sort_values("damage_ratio")[[
    "client_id", "train_n", "damage_ratio", "test_acc", "test_loss"
]])

print("\nAggregate client-level metrics:")
print("mean_client_acc:", last["test_acc"].mean())
print("std_client_acc:", last["test_acc"].std())
print("worst_client_acc:", last["test_acc"].min())
print("best_client_acc:", last["test_acc"].max())

low_damage = last[last["damage_ratio"] < 0.1]
high_damage = last[last["damage_ratio"] > 0.5]

print("\nLow-damage clients avg acc:", low_damage["test_acc"].mean())
print("High-damage clients avg acc:", high_damage["test_acc"].mean())

merged = last.merge(summary[["client_id", "n", "damaged", "damage_ratio"]], on="client_id", suffixes=("", "_summary"))
merged.to_csv("results/xbd_realistic_50r/latest_client_analysis.csv", index=False)

print("\nSaved:")
print("results/xbd_realistic_50r/latest_client_analysis.csv")
