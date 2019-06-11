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

args = get_args()
rng = np.random.RandomState(args.seed)

def evaluate(model, data):
	print("evaluating the model")	

	reporter = report_multistep_error if args.multi_step_prediction else report_singlestep_error

	write_to_file("Errors for all testing data:")
	reporter(data, model)
	write_to_file("\n")
	print("")

	write_to_file("Trying 10 random samples:")
	reporter(data.get_random_samples(10), model)

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

def report_singlestep_error(sample_generator, model):
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

def report_multistep_error(sample_generator, model):
	all_step_losses = []
	ten_step_losses = []

	for predictions, y in iterate_prediction_batches(sample_generator, model):

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

def iterate_prediction_batches(sample_generator, model):
	batch = None
	batch_size = 10000
	for x, y in sample_generator:
		while batch is None or batch[0].shape[0] < batch_size:
			predictions = model.forward(x)
			if batch is None:
				batch = (predictions, y) 
			else:
				batch = (np.concatenate((predictions, batch[0]), axis=0), np.concatenate((y, batch[1]), axis=0))

			# print(f"{batch[0].shape[0]}/{batch_size} batch collected")
		
		assert len(batch[0]) == len(batch[1]) == batch_size

		yield batch
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

