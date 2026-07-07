import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CSV = "datasets/xbd_processed/xbd_crops_metadata.csv"

class XBDDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["post_crop_path"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(int(row["label_binary"]), dtype=torch.long)
        return x, y

df = pd.read_csv(CSV)
df = df.sample(frac=1, random_state=42)

print("Total samples:", len(df))
print("\nLabel counts:")
print(df["label_binary"].value_counts())
print("\nClient counts:")
print(df["client_id"].value_counts().head(20))

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_binary"] if df["label_binary"].nunique() > 1 else None,
)

train_loader = DataLoader(XBDDataset(train_df), batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(XBDDataset(test_df), batch_size=64, shuffle=False, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\ndevice:", device)
print("train:", len(train_df), "test:", len(test_df))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(train_loader, desc=f"epoch {epoch} train"):
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        train_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"epoch {epoch} test"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            pred = logits.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"epoch={epoch} train_loss={train_loss/len(train_df):.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
