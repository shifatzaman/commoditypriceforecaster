# training/reconstruct.py
import numpy as np

def reconstruct_price(last_price, diff_preds):
    """
    last_price: (N,) array
    diff_preds: (N, H) array of predicted log-differences
    """
    prices = []
    for h in range(diff_preds.shape[1]):
        prices.append(last_price * np.exp(diff_preds[:, h]))
    return np.stack(prices, axis=1)