
import numpy as np

def compute_metrics(y_true, y_pred):
    maes = np.mean(np.abs(y_true - y_pred), axis=0)
    return {
        "MAE@1": maes[0],
        "MAE@2": maes[1],
        "MAE@3": maes[2],
        "AvgMAE": maes.mean(),
        "WorstMAE": maes.max()
    }
