import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training.dataset import TimeSeriesDataset
from training.metrics import compute_metrics
from training.train_teacher import train_teacher
from training.reconstruct import reconstruct_price

from models.teachers.dlinear import DLinear
from models.teachers.timemixer import TimeMixer
from models.teachers.patchtst import PatchTST
from models.teachers.pattn import PAttn

# ================= CONFIG =================
LOOKBACK = 48
HORIZONS = [1, 2, 3]
BATCH_SIZE = 32
EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================

def load_series(path):
    return pd.read_csv(path)["price"].values.astype(np.float32)


def split_indices(n, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = list(range(n))
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def predict_prices(model, loader):
    model.eval()
    diff_preds, last_prices = [], []

    with torch.no_grad():
        for x, _, last_price in loader:
            x = x.to(DEVICE)
            diff_preds.append(model(x).cpu().numpy())
            last_prices.append(last_price.numpy())

    diff_preds = np.vstack(diff_preds)
    last_prices = np.concatenate(last_prices)

    return reconstruct_price(last_prices, diff_preds)


def collect_true_prices(series, indices):
    y = []
    for idx in indices:
        y.append([series[idx + LOOKBACK + h] for h in HORIZONS])
    return np.array(y)


def train_and_eval(dataset_name, series):
    ds = TimeSeriesDataset(series, LOOKBACK, HORIZONS)
    tr, va, te = split_indices(len(ds))

    train_loader = DataLoader(Subset(ds, tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(ds, va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(ds, te), batch_size=BATCH_SIZE, shuffle=False)

    y_val  = collect_true_prices(series, va)
    y_test = collect_true_prices(series, te)

    teachers = {
        "DLinear": DLinear,
        "TimeMixer": TimeMixer,
        "PatchTST": PatchTST,
        "PAttn": PAttn,
    }

    rows = []

    for name, cls in teachers.items():
        print(f"Training {name} on {dataset_name}")

        model = cls(LOOKBACK, HORIZONS)
        model = train_teacher(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=1e-3,
            patience=6,
            device=DEVICE,
        )

        pred_val  = predict_prices(model, val_loader)
        pred_test = predict_prices(model, test_loader)

        rows.append({
            "dataset": dataset_name,
            "model": name,
            "split": "val",
            **compute_metrics(y_val, pred_val)
        })
        rows.append({
            "dataset": dataset_name,
            "model": name,
            "split": "test",
            **compute_metrics(y_test, pred_test)
        })

    return rows


def main():
    rice = load_series("data/Wfp_rice.csv")
    wheat = load_series("data/Wfp_wheat.csv")

    rows = []
    rows.extend(train_and_eval("rice", rice))
    rows.extend(train_and_eval("wheat", wheat))

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/results.csv", index=False)

    print("\n=== Test MAE ranking ===")
    print(
        df[df.split == "test"]
        .sort_values("AvgMAE")[["dataset", "model", "AvgMAE"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()