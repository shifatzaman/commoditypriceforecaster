import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training.dataset import TimeSeriesDataset
from training.metrics import compute_metrics
from training.ensemble import ensemble_predictions
from training.distill import distill_student

from models.teachers.patchtst import PatchTST
from models.teachers.dlinear import DLinear
from models.teachers.pattn import PAttn
from models.teachers.timemixer import TimeMixer

from models.students.mlp import MLP
from models.students.kan import KAN

LOOKBACK = 36
HORIZONS = [1,2,3]
BATCH_SIZE = 32

# Training (students)
STUD_EPOCHS = 25
STUD_LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_series(path):
    return pd.read_csv(path)["price"].values.astype(np.float32)

def make_splits(dataset_len, train_ratio=0.7, val_ratio=0.15):
    # sequential split (no leakage)
    n_train = int(dataset_len * train_ratio)
    n_val = int(dataset_len * val_ratio)
    n_test = dataset_len - n_train - n_val
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n_train + n_val + n_test))
    return train_idx, val_idx, test_idx

def predict_model(model, loader):
    model.eval()
    outs = []
    with torch.no_grad():
        for x,_ in loader:
            outs.append(model(x.float()).detach().cpu().numpy())
    return np.vstack(outs)

def evaluate_preds(y_true, y_pred):
    return compute_metrics(y_true, y_pred)

def run_one_combo(dataset_name, series, combo, teachers_map):
    ds = TimeSeriesDataset(series, LOOKBACK, HORIZONS)
    train_idx, val_idx, test_idx = make_splits(len(ds))

    # IMPORTANT: shuffle=False to keep alignment with precomputed teacher ensemble targets in distillation
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    # Ground truth arrays
    y_train = np.vstack([y.numpy() for _,y in train_loader])
    y_val   = np.vstack([y.numpy() for _,y in val_loader])
    y_test  = np.vstack([y.numpy() for _,y in test_loader])

    # -------- Teachers -> Ensemble predictions (per split) --------
    teacher_preds_train = {}
    teacher_preds_val = {}
    teacher_preds_test = {}

    for tname in combo:
        tmodel = teachers_map[tname](LOOKBACK, HORIZONS).to(DEVICE)

        teacher_preds_train[tname] = predict_model(tmodel, train_loader)
        teacher_preds_val[tname]   = predict_model(tmodel, val_loader)
        teacher_preds_test[tname]  = predict_model(tmodel, test_loader)

    ens_train = ensemble_predictions(teacher_preds_train)
    ens_val   = ensemble_predictions(teacher_preds_val)
    ens_test  = ensemble_predictions(teacher_preds_test)

    rows = []

    # Log ensemble metrics (teacher side)
    m_val = evaluate_preds(y_val, ens_val)
    m_test = evaluate_preds(y_test, ens_test)
    rows.append({
        "dataset": dataset_name,
        "teachers": "+".join(combo),
        "model_type": "ensemble",
        "split": "val",
        **m_val
    })
    rows.append({
        "dataset": dataset_name,
        "teachers": "+".join(combo),
        "model_type": "ensemble",
        "split": "test",
        **m_test
    })

    # -------- Distill into students --------
    # MLP student
    mlp = MLP(LOOKBACK, HORIZONS)
    mlp = distill_student(
        mlp,
        train_loader,
        teacher_ens_train=ens_train,
        epochs=STUD_EPOCHS,
        lr=STUD_LR,
        device=DEVICE,
        verbose=False
    ).to(DEVICE)

    mlp_val_pred = predict_model(mlp, val_loader)
    mlp_test_pred = predict_model(mlp, test_loader)

    rows.append({
        "dataset": dataset_name,
        "teachers": "+".join(combo),
        "model_type": "student_mlp",
        "split": "val",
        **evaluate_preds(y_val, mlp_val_pred)
    })
    rows.append({
        "dataset": dataset_name,
        "teachers": "+".join(combo),
        "model_type": "student_mlp",
        "split": "test",
        **evaluate_preds(y_test, mlp_test_pred)
    })

    # KAN student
    kan = KAN(LOOKBACK, HORIZONS)
    kan = distill_student(
        kan,
        train_loader,
        teacher_ens_train=ens_train,
        epochs=STUD_EPOCHS,
        lr=STUD_LR,
        device=DEVICE,
        verbose=False
    ).to(DEVICE)

    kan_val_pred = predict_model(kan, val_loader)
    kan_test_pred = predict_model(kan, test_loader)

    rows.append({
        "dataset": dataset_name,
        "teachers": "+".join(combo),
        "model_type": "student_kan",
        "split": "val",
        **evaluate_preds(y_val, kan_val_pred)
    })
    rows.append({
        "dataset": dataset_name,
        "teachers": "+".join(combo),
        "model_type": "student_kan",
        "split": "test",
        **evaluate_preds(y_test, kan_test_pred)
    })

    return rows

def main():
    rice = load_series("data/Wfp_rice.csv")
    wheat = load_series("data/Wfp_wheat.csv")

    teachers_map = {
        "PatchTST": PatchTST,
        "DLinear": DLinear,
        "PAttn": PAttn,
        "TimeMixer": TimeMixer
    }

    all_rows = []
    all_combos = []
    names = list(teachers_map.keys())
    for r in range(1, len(names)+1):
        all_combos.extend(list(itertools.combinations(names, r)))

    print(f"Running {len(all_combos)} teacher combinations on rice and wheat...")
    for combo in all_combos:
        combo = tuple(combo)
        print("Teachers:", "+".join(combo))

        all_rows.extend(run_one_combo("rice", rice, combo, teachers_map))
        all_rows.extend(run_one_combo("wheat", wheat, combo, teachers_map))

    df = pd.DataFrame(all_rows)

    # Add a joint score per model_type+teachers based on TEST AvgMAE (max across datasets)
    test_df = df[df["split"]=="test"].copy()
    joint = (
        test_df.groupby(["model_type","teachers"])["AvgMAE"]
               .apply(lambda s: float(max(s.values)))  # rice/wheat max
               .reset_index(name="JointScore_test")
    )
    df = df.merge(joint, on=["model_type","teachers"], how="left")

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/results.csv", index=False)
    print("Saved outputs/results.csv")

    # Print top-10 by JointScore_test for students
    top = (df[(df["split"]=="test") & (df["model_type"].isin(["student_mlp","student_kan"]))]
           .drop_duplicates(["model_type","teachers"])
           .sort_values("JointScore_test")
           .head(10))
    print("\nTop student runs by JointScore_test:")
    print(top[["model_type","teachers","JointScore_test"]].to_string(index=False))

if __name__ == "__main__":
    main()
