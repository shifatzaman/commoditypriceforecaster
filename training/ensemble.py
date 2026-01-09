
import numpy as np
def ensemble_predictions(preds):
    return np.mean(np.stack(list(preds.values()), axis=0), axis=0)
