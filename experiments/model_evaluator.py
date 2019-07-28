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

import tensorflow as tf
import keras.backend as K

def get_save_dir():
	if args.model_file == 'none':
		return parent_folder

	return path.dirname(path.dirname(args.model_file))

def get_prediction_save_path(cells):
	name = "predictions.npy" if len(cells) > 0 else "predictions_fullgrid.npy"
	return path.join(get_save_dir(), name)

def write_to_file(message):
	model_folder = get_save_dir()
	file = path.join(model_folder, "evaluation.txt")

	with open(file, "a") as f:
		f.write(message + "\n")

args = get_args()
write_to_file(str(args) + "\n")
rng = np.random.RandomState(args.seed)

def evaluate():
	model, data = get_essentials()

	print("evaluating the model")

	write_to_file("Errors for all testing data:")
	report_multistep_error(data, model, data.num_batches, args.evaluation_steps, args.prediction_batch_size)
	write_to_file("\n")
	print("")

	write_to_file("Trying 10 random samples:")
	report_multistep_error(data.get_random_samples(10), model, "?", 
		args.evaluation_steps, args.prediction_batch_size)

def prediction_analysis():
	model, data = get_essentials()

	indexes = [10, 11, 88, 221]	
	# cells = [(38, 63), (49, 58), (47, 58), (48, 55)]
	cells = []

	results = {}

	save_path = get_prediction_save_path(cells)

	print(f"\nPredictions for {len(indexes)} samples are going to be saved in {save_path}\n")
	for i, batch in enumerate(iterate_prediction_batches(data.enumerate_data(indexes), model, "?", 
		args.prediction_batch_size)):
		print(f"evaluating sample {i}")
		predictions, y = batch

		print(predictions.shape)
		print(y.shape)
		# hacks..
		# checks if predictions are full grid or per cell
		if predictions.shape[0] == args.grid_size ** 2:		
			predictions = predictions.reshape(args.grid_size, args.grid_size, args.output_size)
			predictions = np.transpose(predictions, (2, 0, 1))

			y = y.reshape(args.grid_size, args.grid_size, args.output_size)
			y = np.transpose(y, (2, 0, 1))
		else:
			predictions = np.squeeze(predictions)
			y = np.squeeze(y)
		
		if len(cells) > 0:
			key = lambda indx, cell: f"{indx}_{cell[0]}_{cell[1]}"
			target_key = lambda indx, cell: f"{indx}_{cell[0]}_{cell[1]}_y"
			result_item = { key(indexes[i], cell): predictions[:, cell[0], cell[1]] for cell in cells}
			target_item = { target_key(indexes[i], cell): y[:, cell[0], cell[1]] for cell in cells}
		else:
			result_item = { str(indexes[i]): predictions[-1]}
			target_item = { f"{indexes[i]}_y": y[-1]}


		results.update(result_item)
		results.update(target_item)

	np.save(save_path, results)

def get_essentials():
	print("loading the model")
	model = model_factory.get_model(args)
	# load weights
	if args.model_file != 'none':
		model.load(args.model_file)
		print("model loaded")

	print("getting the data providers")
	test_data = data_provider_factory.get_data_providers(args, rng, test_set=True)


	return model, test_data

def report_multistep_error(sample_generator, model, num_batches, steps, prediction_batch_size):
	steps = list(set(steps))
	losses = defaultdict(list)

	step_check_performed = False
	batches_processed = 0
	for predictions, y in iterate_prediction_batches(sample_generator, model, num_batches, prediction_batch_size):
		if not step_check_performed:
			valid_steps = [st for st in steps if st <= predictions.shape[1]]
			for num_steps in [st for st in steps if not st in valid_steps]:
				print(f"predictions are shorter than {num_steps} steps, skipping")
			step_check_performed = True

			print(f"predictions.shape[0]: {predictions.shape[0]}")

		for num_steps in valid_steps:
			loss = calculate_loss(predictions[:, :num_steps], y[:, :num_steps])
			losses[num_steps].append(loss)

		batches_processed += 1

	print(f"Processed {batches_processed} number of batches")

	for num_steps in losses:
		write_to_file(f"mean {num_steps} step nrmse loss: {np.mean(losses[num_steps])}")
		write_to_file(f"std: {np.std(losses[num_steps])}")

def iterate_prediction_batches(sample_generator, model, num_batches, batch_size):
	batch = None
	batch_count = 0
	for x, y in sample_generator:
		how_much_is_missing(x, y)

		predictions = model.forward(x)

		if batch is None:
			batch = (predictions, y) 
		else:
			batch = (np.concatenate((batch[0], predictions), axis=0), np.concatenate((batch[1], y), axis=0))

		print(f"{batch_count}/{num_batches} processed")
		batch_count += 1

		if batch[0].shape[0] == batch_size:
			yield batch
			batch = None
			continue
		
		assert len(batch[0]) == len(batch[1]) < batch_size, \
			"prediction batch size and batch sizes given by data provider do not match"

def how_much_is_missing(x, y):
	x_dropped = len(x[x == 0])
	y_dropped = len(y[y == 0])
	print(f"x dropped: {x_dropped}, x len: {len(x)}, dropped fraction: {x_dropped / x.size}")
	print(f"y dropped: {y_dropped}, y len: {len(y)}, dropped fraction: {y_dropped / y.size}")

def calculate_loss(predictions, targets):
	predictions = predictions * args.train_std + args.train_mean 
	targets = targets * args.train_std + args.train_mean
	
	return nrmse(targets, predictions)


def profile(operation):
	run_meta = tf.RunMetadata()
	with tf.Session(graph=tf.Graph()) as sess:
		K.set_session(sess)

		model, data = get_essentials()
		return tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=operation)

def evaluate_inference_flops():
	flops = profile(tf.profiler.ProfileOptionBuilder.float_operation())

	print(f"flops: {flops}")
	print(f"flops.total_float_ops: {flops.total_float_ops}")

def evaluate_memory():	
	something = profile(tf.profiler.ProfileOptionBuilder.time_and_memory())

	print(f"something: {something}")
	# print(f"flops.total_float_ops: {flops.total_float_ops}")	



# evaluate()
# evaluate_memory()
prediction_analysis()

