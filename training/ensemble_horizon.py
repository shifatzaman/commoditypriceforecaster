# training/ensemble_horizon.py
import numpy as np

def compute_horizon_weights_from_val(preds_by_teacher_val, y_val, eps=1e-8):
    """
    preds_by_teacher_val: dict[name] -> (N, H) price predictions on VAL
    y_val: (N, H) true prices on VAL
    Returns:
      weights: (K, H) where weights[:,h] sums to 1
      teachers: list of teacher names aligned to weights axis 0
      maes: (K, H) MAE per teacher per horizon
    """
    teachers = list(preds_by_teacher_val.keys())
    K = len(teachers)
    H = y_val.shape[1]

    maes = np.zeros((K, H), dtype=np.float64)
    for i, t in enumerate(teachers):
        maes[i] = np.mean(np.abs(y_val - preds_by_teacher_val[t]), axis=0)

    inv = 1.0 / (maes + eps)
    weights = inv / np.sum(inv, axis=0, keepdims=True)
    return weights, teachers, maes

def apply_horizon_ensemble(preds_by_teacher, weights, teachers):
    """
    preds_by_teacher: dict[name] -> (N, H) price predictions
    weights: (K, H)
    teachers: list of K teacher names matching weights
    Returns:
      ens_pred: (N, H)
    """
    stacked = np.stack([preds_by_teacher[t] for t in teachers], axis=0)  # (K, N, H)
    # weighted sum across teachers per horizon
    ens = np.sum(stacked * weights[:, None, :], axis=0)  # (N, H)
    return ens