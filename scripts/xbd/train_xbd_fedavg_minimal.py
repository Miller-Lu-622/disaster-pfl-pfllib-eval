import copy
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

TRAIN_CSV = "datasets/xbd_processed/fl_manifest_balanced/train_manifest.csv"
TEST_CSV = "datasets/xbd_processed/fl_manifest_balanced/test_manifest.csv"

NUM_CLIENTS = 5
ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LR = 1e-3

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

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)

def train_one_client(global_model, loader, device):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total = 0

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

    return model.cpu().state_dict(), total, total_loss / max(total, 1)

def fedavg(client_states, client_sizes):
    total = sum(client_sizes)
    avg = copy.deepcopy(client_states[0])

    for k in avg.keys():
        avg[k] = avg[k].float() * (client_sizes[0] / total)
        for i in range(1, len(client_states)):
            avg[k] += client_states[i][k].float() * (client_sizes[i] / total)

    return avg

def evaluate(model, loader, device):
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)

    return correct / max(total, 1), loss_sum / max(total, 1)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    client_counts = train_df["client_id"].value_counts()
    clients = list(client_counts.head(NUM_CLIENTS).index)

    print("selected clients:")
    for c in clients:
        g = train_df[train_df["client_id"] == c]
        print(c, "n=", len(g), "damage_ratio=", g["label_binary"].mean())

    train_loaders = {}
    test_loaders = {}

    for c in clients:
        c_train = train_df[train_df["client_id"] == c]
        c_test = test_df[test_df["client_id"] == c]

        train_loaders[c] = DataLoader(
            XBDDataset(c_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )

        test_loaders[c] = DataLoader(
            XBDDataset(c_test),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
        )

    all_test = test_df[test_df["client_id"].isin(clients)]
    global_test_loader = DataLoader(
        XBDDataset(all_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    global_model = SmallCNN()

    for r in range(ROUNDS):
        print(f"\n===== ROUND {r} =====")

        client_states = []
        client_sizes = []

        for c in clients:
            state, size, train_loss = train_one_client(global_model, train_loaders[c], device)
            client_states.append(state)
            client_sizes.append(size)
            print(f"client={c} train_n={size} train_loss={train_loss:.4f}")

        avg_state = fedavg(client_states, client_sizes)
        global_model.load_state_dict(avg_state)

        global_acc, global_loss = evaluate(global_model, global_test_loader, device)
        print(f"round={r} global_test_acc={global_acc:.4f} global_test_loss={global_loss:.4f}")

        for c in clients:
            acc, loss = evaluate(global_model, test_loaders[c], device)
            print(f"  eval_client={c} acc={acc:.4f} loss={loss:.4f}")

    print("\nDONE minimal FedAvg xBD")

if __name__ == "__main__":
    main()
