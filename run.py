
import itertools, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from training.dataset import TimeSeriesDataset
from training.metrics import compute_metrics
from training.ensemble import ensemble_predictions
from models.teachers.patchtst import PatchTST
from models.teachers.dlinear import DLinear
from models.teachers.pattn import PAttn
from models.teachers.timemixer import TimeMixer

LOOKBACK = 36
HORIZONS = [1,2,3]

def load_series(path):
    return pd.read_csv(path)["price"].values.astype(np.float32)

def run_dataset(name, series):
    dataset = TimeSeriesDataset(series, LOOKBACK, HORIZONS)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    teachers = {
        "PatchTST": PatchTST,
        "DLinear": DLinear,
        "PAttn": PAttn,
        "TimeMixer": TimeMixer
    }

    rows = []
    for r in range(1, len(teachers)+1):
        for combo in itertools.combinations(teachers.keys(), r):
            preds = {}
            for tname in combo:
                model = teachers[tname](LOOKBACK, HORIZONS)
                outs = []
                for x,_ in loader:
                    outs.append(model(x).detach().numpy())
                preds[tname] = np.vstack(outs)

            ens = ensemble_predictions(preds)
            y_true = np.vstack([y.numpy() for _,y in loader])
            metrics = compute_metrics(y_true, ens)
            metrics.update({"dataset": name, "teachers": "+".join(combo)})
            rows.append(metrics)
    return rows

def main():
    rice = load_series("data/Wfp_rice.csv")
    wheat = load_series("data/Wfp_wheat.csv")

    rows = []
    rows.extend(run_dataset("rice", rice))
    rows.extend(run_dataset("wheat", wheat))

    df = pd.DataFrame(rows)
    df.to_csv("outputs/results.csv", index=False)
    print("Saved outputs/results.csv")

if __name__ == "__main__":
    main()
