import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training.dataset import TimeSeriesDataset
from training.metrics import compute_metrics
from training.train_teacher import train_teacher
from training.reconstruct import reconstruct_price
from training.ensemble_horizon import (
    compute_horizon_weights_from_val,
    apply_horizon_ensemble,
)
from training.distill_safe import distill_student_safe

from models.teachers.dlinear import DLinear
from models.teachers.timemixer import TimeMixer
from models.teachers.patchtst import PatchTST
from models.teachers.pattn import PAttn

from models.students.mlp import MLP
from models.students.kan import KAN

# ================= CONFIG =================
LOOKBACK = 48
HORIZONS = [1, 2, 3]
BATCH_SIZE = 32
EPOCHS_TEACHER = 40
EPOCHS_STUDENT = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

def load_series(path):
    return pd.read_csv(path)["price"].values.astype(np.float32)

def split_indices(n, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = list(range(n))
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def collect_true_prices(series, indices):
    y = []
    for idx in indices:
        y.append([series[idx + LOOKBACK + h] for h in HORIZONS])
    return np.array(y, dtype=np.float32)

def predict_diff(model, loader):
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _, _last_price in loader:
            x = x.to(DEVICE)
            outs.append(model(x).cpu().numpy())
    return np.vstack(outs).astype(np.float32)

def predict_prices_from_diff(model, loader):
    model.eval()
    diff_preds = []
    last_prices = []
    with torch.no_grad():
        for x, _, last_price in loader:
            x = x.to(DEVICE)
            diff_preds.append(model(x).cpu().numpy())
            last_prices.append(last_price.numpy())
    diff_preds = np.vstack(diff_preds).astype(np.float32)
    last_prices = np.concatenate(last_prices).astype(np.float32)
    return reconstruct_price(last_prices, diff_preds).astype(np.float32)

def run_dataset(dataset_name, series):
    ds = TimeSeriesDataset(series, LOOKBACK, HORIZONS)
    tr, va, te = split_indices(len(ds))

    # Teacher training loader can shuffle; KD loader should not shuffle.
    train_loader_teacher = DataLoader(Subset(ds, tr), batch_size=BATCH_SIZE, shuffle=True)
    train_loader_kd      = DataLoader(Subset(ds, tr), batch_size=BATCH_SIZE, shuffle=False)

    val_loader  = DataLoader(Subset(ds, va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(ds, te), batch_size=BATCH_SIZE, shuffle=False)

    y_val_prices  = collect_true_prices(series, va)
    y_test_prices = collect_true_prices(series, te)

    teachers = {
        "DLinear": DLinear,
        "TimeMixer": TimeMixer,
        "PatchTST": PatchTST,
        "PAttn": PAttn,
    }

    rows = []

    trained = {}
    preds_val_prices = {}
    preds_test_prices = {}

    # ---------------- Train + eval each teacher ----------------
    for name, cls in teachers.items():
        print(f"Training {name} on {dataset_name}")
        model = cls(LOOKBACK, HORIZONS)

        model = train_teacher(
            model,
            train_loader_teacher,
            val_loader,
            epochs=EPOCHS_TEACHER,
            lr=1e-3,
            patience=6,
            device=DEVICE,
        )
        trained[name] = model

        pv = predict_prices_from_diff(model, val_loader)
        pt = predict_prices_from_diff(model, test_loader)

        preds_val_prices[name] = pv
        preds_test_prices[name] = pt

        rows.append({
            "dataset": dataset_name,
            "model_type": "teacher",
            "model": name,
            "split": "val",
            **compute_metrics(y_val_prices, pv)
        })
        rows.append({
            "dataset": dataset_name,
            "model_type": "teacher",
            "model": name,
            "split": "test",
            **compute_metrics(y_test_prices, pt)
        })

    # ---------------- Per-horizon ensemble (weights from VAL) ----------------
    weights, teacher_order, maes = compute_horizon_weights_from_val(preds_val_prices, y_val_prices)

    ens_val = apply_horizon_ensemble(preds_val_prices, weights, teacher_order)
    ens_test = apply_horizon_ensemble(preds_test_prices, weights, teacher_order)

    rows.append({
        "dataset": dataset_name,
        "model_type": "ensemble_horizon",
        "model": "+".join(teacher_order),
        "split": "val",
        "weights": str(weights.tolist()),
        **compute_metrics(y_val_prices, ens_val)
    })
    rows.append({
        "dataset": dataset_name,
        "model_type": "ensemble_horizon",
        "model": "+".join(teacher_order),
        "split": "test",
        "weights": str(weights.tolist()),
        **compute_metrics(y_test_prices, ens_test)
    })

    # ---------------- KD targets: ensemble in DIFF space ----------------
    # Build teacher diff preds for the train split for each teacher, then ensemble per horizon.
    preds_train_diff = {}
    for name, model in trained.items():
        preds_train_diff[name] = predict_diff(model, train_loader_kd)  # (N_train, H)

    # Apply the SAME per-horizon weights in diff space
    ens_train_diff = apply_horizon_ensemble(preds_train_diff, weights, teacher_order)  # (N_train, H)

    # ---------------- Distill students (MLP + KAN) ----------------
    print(f"Distilling students on {dataset_name} (from per-horizon ensemble)")

    mlp = MLP(LOOKBACK, HORIZONS)
    mlp = distill_student_safe(
        mlp,
        train_loader_kd,
        teacher_train_targets_diff=ens_train_diff,
        epochs=EPOCHS_STUDENT,
        lr=1e-3,
        device=DEVICE,
    )

    kan = KAN(LOOKBACK, HORIZONS)
    kan = distill_student_safe(
        kan,
        train_loader_kd,
        teacher_train_targets_diff=ens_train_diff,
        epochs=EPOCHS_STUDENT,
        lr=1e-3,
        device=DEVICE,
    )

    # Evaluate students in PRICE space
    mlp_test_prices = predict_prices_from_diff(mlp.to(DEVICE), test_loader)
    kan_test_prices = predict_prices_from_diff(kan.to(DEVICE), test_loader)

    rows.append({
        "dataset": dataset_name,
        "model_type": "student_kd",
        "model": "MLP",
        "split": "test",
        **compute_metrics(y_test_prices, mlp_test_prices)
    })
    rows.append({
        "dataset": dataset_name,
        "model_type": "student_kd",
        "model": "KAN",
        "split": "test",
        **compute_metrics(y_test_prices, kan_test_prices)
    })

    return rows

def main():
    rice = load_series("data/Wfp_rice.csv")
    wheat = load_series("data/Wfp_wheat.csv")

    rows = []
    rows.extend(run_dataset("rice", rice))
    rows.extend(run_dataset("wheat", wheat))

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/results.csv", index=False)

    print("\n=== TEST ranking (AvgMAE) ===")
    print(df[df["split"]=="test"].sort_values("AvgMAE")[["dataset","model_type","model","AvgMAE"]].to_string(index=False))

if __name__ == "__main__":
    main()