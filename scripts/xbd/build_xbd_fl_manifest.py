import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

IN = Path("datasets/xbd_processed/xbd_crops_metadata.csv")
OUT = Path("datasets/xbd_processed/fl_manifest_realistic")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

# 保留样本数足够的 clients
counts = df["client_id"].value_counts()
valid_clients = counts[counts >= 30].index
df = df[df["client_id"].isin(valid_clients)].copy()

print("valid clients:", df["client_id"].nunique())
print("total samples after filter:", len(df))

train_rows = []
test_rows = []

for client_id, g in df.groupby("client_id"):
    stratify = g["label_binary"] if g["label_binary"].nunique() > 1 else None

    train, test = train_test_split(
        g,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
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

print("\ntrain:", train_df.shape)
print("test:", test_df.shape)
print("\nclient summary:")
print(summary)
