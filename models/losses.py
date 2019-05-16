import numpy as np

def l1_loss(predictions, targets):
	return np.abs(targets - predictions).sum()