import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from os import path
import sys
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_folder)

from arg_extractor import get_args
from models.losses import nrmse_numpy as nrmse
import model_factory
from data_providers import data_provider_factory

import numpy as np
from collections import defaultdict

args = get_args()
rng = np.random.RandomState(args.seed)

def evaluate(model, data):
	print("evaluating the model")

	write_to_file("Errors for all testing data:")
	report_multistep_error(data, model, data.num_batches, args.evaluation_steps, args.prediction_batch_size)
	write_to_file("\n")
	print("")

	write_to_file("Trying 10 random samples:")
	report_multistep_error(data.get_random_samples(10), model, "?", 
		args.evaluation_steps, args.prediction_batch_size)

def prediction_analysis(model, data):
	indexes = [0, 25, 50, 75]	
	results = []

	print(f"\nPredictions for {len(indexes)} samples are going to be saved in preditions.npy\n")
	for i, batch in enumarate(iterate_prediction_batches(data.enumerate_data(indexes))):
		print(f"evaluating sample {i}")
		predictions, y = batch

		result_item = {
			'input': x,
			'targets': y,
			'predictions': predictions
		}

		results.append(result_item)

	np.save("predictions.npy", results)

def get_essentials():
	print("loading the model")
	model = model_factory.get_model(args)
	# load weights
	model.load(args.model_file)
	print("model loaded")

	print("getting the data providers")
	test_data = data_provider_factory.get_data_providers(args, rng, test_set=True)


	return model, test_data

def report_multistep_error(sample_generator, model, num_batches, steps, prediction_batch_size):
	steps = list(set(steps))
	losses = defaultdict(list)

	step_check_performed = False
	for predictions, y in iterate_prediction_batches(sample_generator, model, num_batches, prediction_batch_size):
		if not step_check_performed:
			valid_steps = [st for st in steps if st <= predictions.shape[1]]
			for num_steps in [st for st in steps if not st in valid_steps]:
				print(f"predictions are shorter than {num_steps} steps, skipping")
			step_check_performed = True

		for num_steps in valid_steps:
			loss = calculate_loss(predictions[:, :num_steps], y[:, :num_steps])
			losses[num_steps].append(loss)

	for num_steps in losses:
		write_to_file(f"mean {num_steps} step nrmse loss: {np.mean(losses[num_steps])}")
		write_to_file(f"std: {np.std(losses[num_steps])}")

def iterate_prediction_batches(sample_generator, model, num_batches, batch_size):
	batch = None
	batch_count = 0
	for x, y in sample_generator:
		if batch is None or batch[0].shape[0] < batch_size:
			predictions = model.forward(x)
			print(f"PREDICTIONS: {predictions}")
			if batch is None:
				batch = (predictions, y) 
			else:
				batch = (np.concatenate((predictions, batch[0]), axis=0), np.concatenate((y, batch[1]), axis=0))

			batch_count += 1
			continue
		
		assert len(batch[0]) == len(batch[1]) == batch_size, \
			"prediction batch size and batch sizes given by data provider do not match"

		yield batch
		print(f"{batch_count}/{num_batches} processed")
		batch = None

def calculate_loss(predictions, targets):
	predictions = predictions * args.train_std + args.train_mean 
	targets = targets * args.train_std + args.train_mean
	
	return nrmse(targets, predictions)
	

def write_to_file(message):
	model_folder = path.dirname(path.dirname(args.model_file))
	file = path.join(model_folder, "evaluation.txt")

	with open(file, "a") as f:
		f.write(message + "\n")


model, data = get_essentials()
evaluate(model, data)
# prediction_analysis(model, data)

