
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
from os import path
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_folder)
import numpy as np
import time
from arg_extractor import get_args
import tensorflow as tf

from experiment_builder import ExperimentBuilder
from models.lstm_tf import LSTM
from session_holder import SessionHolder

from data_providers.windowed_data_provider import WindowedDataProvider
from data_providers.data_reader import FullDataReader
from data_providers.data_reader import MiniDataReader


args = get_args()
rng = np.random.RandomState(args.seed)

data_reader = MiniDataReader if args.use_mini_data else FullDataReader

session_holder = SessionHolder()

experiment_builder = ExperimentBuilder(
	args = args,
	model = LSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size,
		session_holder=session_holder, learning_rate=args.learning_rate), 
	experiment_name = args.experiment_name,
	num_epochs = args.num_epochs,
	train_data = WindowedDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='train'), 
			window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
			shuffle_order=True, rng=rng, fraction_of_data=args.fraction_of_data),
	val_data = WindowedDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='valid'), 
			window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
			shuffle_order=True, rng=rng),
	continue_from_epoch=args.continue_from_epoch
)

with tf.Session() as sess:
	session_holder.set_sess(sess)
	experiment_builder.run_experiment()