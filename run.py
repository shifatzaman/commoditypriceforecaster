# run.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training.dataset import TimeSeriesDataset
from training.metrics import compute_metrics
from training.train_teacher import train_teacher

from models.teachers.dlinear import DLinear
from models.students.mlp import MLP

LOOKBACK = 36
HORIZONS = [1, 2, 3]
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_series(path):
    return pd.read_csv(path)["price"].values.astype(np.float32)


def split_indices(n, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = list(range(n))
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def predict(model, loader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in loader:
            outputs.append(model(x.to(DEVICE)).cpu().numpy())
    return np.vstack(outputs)


def run_dataset(name, series):
    dataset = TimeSeriesDataset(series, LOOKBACK, HORIZONS)
    tr, va, te = split_indices(len(dataset))

    train_loader = DataLoader(Subset(dataset, tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(dataset, te), batch_size=BATCH_SIZE, shuffle=False)

    y_val  = np.vstack([y.numpy() for _, y in val_loader])
    y_test = np.vstack([y.numpy() for _, y in test_loader])

    rows = []

    # ---------- Train DLinear teacher ----------
    teacher = DLinear(LOOKBACK, HORIZONS)
    teacher = train_teacher(teacher, train_loader, val_loader)

    pred_val  = predict(teacher, val_loader)
    pred_test = predict(teacher, test_loader)

    rows.append({
        "dataset": name,
        "model_type": "teacher_dlinear_val",
        **compute_metrics(y_val, pred_val)
    })
    rows.append({
        "dataset": name,
        "model_type": "teacher_dlinear_test",
        **compute_metrics(y_test, pred_test)
    })

    # ---------- Distill into MLP ----------
    teacher_train_preds = predict(teacher, train_loader)

    student = MLP(LOOKBACK, HORIZONS).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(20):
        student.train()
        offset = 0
        for x, y in train_loader:
            b = x.shape[0]
            x, y = x.to(DEVICE), y.to(DEVICE)
            t = torch.tensor(teacher_train_preds[offset:offset + b], device=DEVICE)
            offset += b

            pred = student(x)
            loss = loss_fn(pred, y) + 0.3 * loss_fn(pred, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student_test_preds = predict(student, test_loader)

    rows.append({
        "dataset": name,
        "model_type": "student_mlp_test",
        **compute_metrics(y_test, student_test_preds)
    })

    return rows


def main():
    rice = load_series("data/Wfp_rice.csv")
    wheat = load_series("data/Wfp_wheat.csv")

    results = []
    results.extend(run_dataset("rice", rice))
    results.extend(run_dataset("wheat", wheat))

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(results).to_csv("outputs/results.csv", index=False)
    print("Saved outputs/results.csv")


if __name__ == "__main__":
    main()