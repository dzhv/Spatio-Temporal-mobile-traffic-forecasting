import numpy as np

import tensorflow.keras.backend as K

# implementation of normalised root mean square error using Keras operations
def nrmse_keras(targets, predictions):	
	error = predictions - targets
	mse = K.mean(K.square(error))	
	rmse = K.sqrt(mse)
	return rmse / K.mean(targets)
	

def nrmse_numpy(targets, predictions):
	targets = np.squeeze(targets)
	predictions = np.squeeze(predictions)
	
	error = predictions - targets	
	mse = np.mean(np.square(error))
	rmse = np.sqrt(mse)
	return rmse / np.mean(targets)

def mse(targets, predictions):
	targets = np.squeeze(targets)
	predictions = np.squeeze(predictions)

	error = predictions - targets
	return np.mean(np.square(error))
