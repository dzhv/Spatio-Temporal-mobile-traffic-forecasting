import os
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import time

def train_and_save():
	print(f"starting")

	if not os.path.exists("results/arima"):
		os.mkdir("results/arima")
	if not os.path.exists("results/arima/saved_models"):
		os.mkdir("results/arima/saved_models")

	p, d, q = 12, 1, 1

	print(f"loading data")
	train = np.load("data/train.npy")

	window_size = train.shape[-1]

	# fitted_models = []
	total_time = 0
	for i in range(window_size**2):
		start_time = time.time()

		print(f"training model: {i+1}/{window_size**2 + 1}")
		x_coord = i // window_size
		y_coord = i % window_size

		model = ARIMA(train[:, x_coord, y_coord], order=(p, d, q))
		model_fit = model.fit(disp=0)
		# fitted_models.append(model_fit)
		model_fit.save(f"results/arima/save_{x_coord}_{y_coord}.pickle")
		
		elapsed = time.time() - start_time
		total_time += elapsed

		if i % 10 == 0:
			print(f"total time elapsed: {total_time}")
		# print(f"took: {elapsed} seconds")

train_and_save()