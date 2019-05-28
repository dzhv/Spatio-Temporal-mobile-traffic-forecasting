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
from data_providers.seq2seq_data_provider import Seq2SeqDataProvider
from models.losses import nrmse_numpy as nrmse
from models.lstm import LSTM
from models.lstm_seq2seq import LstmSeq2Seq
from models.keras_seq2seq import KerasSeq2Seq

import numpy as np

args = get_args()

def evaluate():
	model, data = get_essentials()

	print("Errors for all testing data")
	print("---------------------------")
	report_error(data, model)
	print("\n")

	print("Trying 10 random samples:")
	print("---------------------------")
	report_error(data.get_random_samples(10), model)

def prediction_analysis():
	model, data = get_essentials()

	indexes = [0, 25, 50, 75]	
	results = []

	print(f"\nPredictions for {len(indexes)} samples are going to be saved in preditions.npy\n")
	for i, batch in enumerate(data.enumerate_data(indexes)):
		print(f"evaluating sample {i}")
		x, y = batch
		predictions = model.forward(x)

		result_item = {
			'targets': y,
			'predictions': predictions
		}

		results.append(result_item)

	np.save("predictions.npy", results)

def get_essentials():
	model = model_factory(args.model_name, args.model_file, args.batch_size)
	print("model loaded")
	
	data_provider = WindowedDataProvider if args.model_name == "lstm" else Seq2SeqDataProvider
	data = data_provider(data_reader = FullDataReader(data_folder=args.data_path, which_set='test'),
		window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
		shuffle_order=False)

	return model, data

def report_error(sample_generator, model):
	losses = []
	for x, y in sample_generator:
		predictions = model.forward(x)
		loss = calculate_loss(predictions, y)
		losses.append(loss)

		print(loss)
		print(f"mean: {np.mean(losses)}")

	print(f"std: {np.std(losses)}")

def calculate_loss(predictions, targets):
	predictions = predictions * args.train_std + args.train_mean 
	targets = targets * args.train_std + args.train_mean
	
	return nrmse(targets, predictions)

def model_factory(model_name, model_file, batch_size):
	model_name = model_name.lower()

	print(f"loading model: {model_name} from {args.model_file}")

	if model_name == "lstm":
		model = LSTM(batch_size=args.batch_size, num_layers=args.num_layers, hidden_size=args.hidden_size)
	elif model_name == "seq2seq":
		model = LstmSeq2Seq(batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size)
	elif model_name == "keras_seq2seq":
		model = KerasSeq2Seq(batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size)
	else:
		raise ValueError(f"unknown model: {model_name}")

	model.load(model_file)
	return model

evaluate()
# prediction_analysis()

