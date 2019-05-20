import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.keras.models import load_model
from arg_extractor import get_args
from data_providers.data_reader import FullDataReader
from data_providers.windowed_data_provider import WindowedDataProvider
from models.losses import nrmse_numpy as nrmse
from models.lstm import LSTM

import numpy as np

args = get_args()

def calculate_loss(predictions, targets):
	predictions = predictions * args.train_std + args.train_mean 
	targets = targets * args.train_std + args.train_mean
	return nrmse(targets, predictions)

def evaluate():	
	print(f"loading model: {args.model_file}")
	model = LSTM(model=load_model(args.model_file), batch_size=args.batch_size)
	print("model loaded")
	
	data = WindowedDataProvider(data_reader = FullDataReader(data_folder=args.data_path, which_set='test'), 
		window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
		shuffle_order=False)

	losses = []
	i = 0
	for x, y in data:
		predictions = model.forward(x)
		loss = calculate_loss(predictions, y)
		losses.append(loss)

		print(loss)
		print(f"mean: {np.array(losses).mean()}")

	print(losses)

evaluate()

