import numpy as np
import keras.backend as K


def l1(targets, predictions):
	return np.abs(targets - predictions).sum()

# implementation using Keras operations
def nrmse_keras(targets, predictions):	
	error = predictions - targets	
	mse = K.mean(K.square(error))
	rmse = K.sqrt(mse)
	return rmse / K.mean(targets)

def nrmse_numpy(targets, predictions):	
	error = predictions - targets	
	mse = np.mean(np.square(error))
	rmse = np.sqrt(mse)
	return rmse / np.mean(targets)