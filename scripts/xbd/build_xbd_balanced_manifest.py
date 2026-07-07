import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

IN = Path("datasets/xbd_processed/xbd_crops_metadata.csv")
OUT = Path("datasets/xbd_processed/fl_manifest_balanced")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

rows = []

# 每个 client 内部尽量平衡 no-damage/damaged
for client_id, g in df.groupby("client_id"):
    neg = g[g["label_binary"] == 0]
    pos = g[g["label_binary"] == 1]

    # 太小的 client 跳过
    if len(pos) < 5 or len(neg) < 5:
        continue

    n = min(len(pos), len(neg), 80)

    sampled = pd.concat([
        neg.sample(n=n, random_state=42),
        pos.sample(n=n, random_state=42),
    ])

    rows.append(sampled)

balanced = pd.concat(rows).sample(frac=1, random_state=42)

train_rows = []
test_rows = []

for client_id, g in balanced.groupby("client_id"):
    train, test = train_test_split(
        g,
        test_size=0.2,
        random_state=42,
        stratify=g["label_binary"],
    )
    train_rows.append(train)
    test_rows.append(test)

train_df = pd.concat(train_rows)
test_df = pd.concat(test_rows)

train_df.to_csv(OUT / "train_manifest.csv", index=False)
test_df.to_csv(OUT / "test_manifest.csv", index=False)

summary = train_df.groupby("client_id").agg(
    n=("sample_id", "count"),
    damaged=("label_binary", "sum"),
    damage_ratio=("label_binary", "mean"),
).sort_values("n", ascending=False)

summary.to_csv(OUT / "client_summary.csv")

print("balanced clients:", train_df["client_id"].nunique())
print("train:", train_df.shape)
print("test:", test_df.shape)
print(summary)
