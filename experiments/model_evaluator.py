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
	model, data, multi_step_prediction = get_essentials()

	reporter = report_multistep_error if multi_step_prediction else report_singlestep_error

	write_to_file("Errors for all testing data:")
	reporter(data, model, multi_step_prediction)
	write_to_file("\n")
	print("")

	write_to_file("Trying 10 random samples:")
	reporter(data.get_random_samples(10), model, multi_step_prediction)

def prediction_analysis():
	model, data, multi_step_prediction = get_essentials()

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
	model, multi_step_prediction = model_factory(args.model_name, args.model_file, args.batch_size)
	print("model loaded")
	
	data_provider = Seq2SeqDataProvider if multi_step_prediction else WindowedDataProvider
	data = data_provider(data_reader = FullDataReader(data_folder=args.data_path, which_set='test'),
		window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
		shuffle_order=False, fraction_of_data=args.fraction_of_data)

	return model, data, multi_step_prediction

def report_singlestep_error(sample_generator, model, num_prediction_steps):
	losses = []
	for x, y in sample_generator:
		predictions = model.forward(x)
		loss = calculate_loss(predictions, y)
		losses.append(loss)

		print(loss)
		print(f"mean: {np.mean(losses)}")

	print(f"std: {np.std(losses)}")

	write_to_file(f"mean nrmse loss: {np.mean(losses)}")
	write_to_file(f"std: {np.std(losses)}")

def report_multistep_error(sample_generator, model, num_prediction_steps):
	all_step_losses = []
	ten_step_losses = []
	for x, y in sample_generator:
		predictions = model.forward(x)
		all_step_loss = calculate_loss(predictions, y)
		all_step_losses.append(all_step_loss)

		ten_step_loss = calculate_loss(predictions[:, :10], y[:, :10])
		ten_step_losses.append(ten_step_loss)

		print(f"all step loss: {all_step_loss}")
		print(f"mean: {np.mean(all_step_losses)}")
		print(f"10 step loss: {ten_step_loss}")
		print(f"mean: {np.mean(ten_step_losses)}")
		

	print(f"all step loss std: {np.std(all_step_losses)}")
	print(f"10 step loss std: {np.std(ten_step_losses)}")

	write_to_file(f"mean all step nrmse loss: {np.mean(all_step_losses)}")
	write_to_file(f"std: {np.std(all_step_losses)}")
	write_to_file(f"mean 10 step nrmse loss: {np.mean(ten_step_losses)}")
	write_to_file(f"std: {np.std(ten_step_losses)}")


def calculate_loss(predictions, targets):
	predictions = predictions * args.train_std + args.train_mean 
	targets = targets * args.train_std + args.train_mean
	
	return nrmse(targets, predictions)

def model_factory(model_name, model_file, batch_size):
	model_name = model_name.lower()

	print(f"loading model: {model_name} from {model_file}")

	multi_step_prediction = True
	if model_name == "lstm":
		model = LSTM(batch_size=args.batch_size, num_layers=args.num_layers, hidden_size=args.hidden_size)
		multi_step_prediction = False
	elif model_name == "seq2seq":
		model = LstmSeq2Seq(batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size)
	elif model_name == "keras_seq2seq":
		model = KerasSeq2Seq(batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size)
	else:
		raise ValueError(f"unknown model: {model_name}")

	model.load(model_file)
	return model, multi_step_prediction

def write_to_file(message):
	model_folder = path.dirname(path.dirname(args.model_file))
	file = path.join(model_folder, "evaluation.txt")

	with open(file, "a") as f:
		f.write(message + "\n")

evaluate()
# prediction_analysis()

