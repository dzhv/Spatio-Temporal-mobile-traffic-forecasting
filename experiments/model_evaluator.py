import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from os import path
import sys
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_folder)

from tensorflow.keras.models import load_model
from arg_extractor import get_args
from data_providers.data_reader import FullDataReader
from data_providers.windowed_data_provider import WindowedDataProvider
from models.losses import nrmse_numpy as nrmse
from models.lstm import LSTM
from models.lstm_seq2seq import LstmSeq2Seq

import numpy as np

args = get_args()

def calculate_loss(predictions, targets):
	predictions = predictions * args.train_std + args.train_mean 
	targets = targets * args.train_std + args.train_mean

	if len(targets.shape) == 1 or targets.shape[-1] == 1:     # if this is a 1 step prediction
		return nrmse(targets, predictions)

	# if this is a multi step prediction
	return nrmse(targets[:, -1], predictions[:, -1])

def get_essentials():
	model = model_factory(args.model_name, args.model_file, args.batch_size)
	print("model loaded")
	
	data = WindowedDataProvider(data_reader = FullDataReader(data_folder=args.data_path, which_set='test'), 
		window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
		shuffle_order=False)

	return model, data

def evaluate():
	model, data = get_essentials()

	print("Errors for all testing data")
	print("---------------------------")
	report_error(data, model)
	print("\n")

	print("Trying 10 random samples:")
	print("---------------------------")
	report_error(data.get_random_samples(10), model)

def report_error(sample_generator, model):
	losses = []
	for x, y in sample_generator:
		predictions = model.forward(x)
		loss = calculate_loss(predictions, y)
		losses.append(loss)

		print(loss)
		print(f"mean: {np.mean(losses)}")

	print(f"std: {np.std(losses)}")

def prediction_analysis():
	model, data = get_essentials()

	indexes = [0, 25, 50, 75]	
	results = []

	print(f"\nPredictions for {len(indexes)} samples are going to be saved in results.npy\n")
	for i, batch in enumerate(data.enumerate_data(indexes)):
		print(f"evaluating sample {i}")
		x, y = batch
		predictions = model.forward(x)

		result_item = {
			'targets': y,
			'predictions': predictions
		}

		results.append(result_item)

	np.save("results.npy", results)


def model_factory(model_name, model_file, batch_size):
	model_name = model_name.lower()

	print(f"loading model: {model_name} from {args.model_file}")

	if model_name == "lstm":
		model = LSTM(batch_size=args.batch_size)
	elif model_name == "seq2seq":
		model = LstmSeq2Seq(batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size,
		learning_rate=args.learning_rate)
	else:
		raise ValueError(f"unknown model: {model_name}")

	model.load(model_file)
	return model

evaluate()
prediction_analysis()

