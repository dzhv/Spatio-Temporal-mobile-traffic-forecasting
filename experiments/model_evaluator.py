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

def write_to_file(message):
	model_folder = path.dirname(path.dirname(args.model_file))
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
			batch = (np.concatenate((predictions, batch[0]), axis=0), np.concatenate((y, batch[1]), axis=0))

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
	graph = tf.Graph()
	with tf.Session(graph=graph) as sess:
		K.set_session(sess)

		
		model, data = get_essentials()
		model.forward(np.random.randn(1, args.segment_size, 11, 11))
		# 1670059  1671066
		# 1671279  1669735
		# print("\nhenlo")
		# for op in graph.get_operations():
		# 	print(str(op.name))
		# print("henlo\n")

		# writer = tf.summary.FileWriter(logdir='logs/test', graph=graph, session=sess)
		# writer.add_run_metadata(run_meta, "meta_tag?")
		# writer.flush()

		# hmm = model.forward(np.random.randn(1, 12, 11, 11))
		# print(hmm)

		# return None
		return tf.profiler.profile(graph, run_meta=run_meta, cmd='op', options=operation)

def evaluate_inference_flops():
	flops = profile(tf.profiler.ProfileOptionBuilder.float_operation())

	# print(f"flops: {flops}")
	print(f"flops.total_float_ops: {flops.total_float_ops}")

def evaluate_memory():	
	something = profile(tf.profiler.ProfileOptionBuilder.time_and_memory())

	print(f"something: {something}")
	# print(f"flops.total_float_ops: {flops.total_float_ops}")	

def count_parameters():
    model, data = get_essentials()

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters    
    print(f"number of parameters: {total_parameters}")
    return total_parameters


# count_parameters()
evaluate_inference_flops()
# evaluate_memory()
# prediction_analysis(model, data)

