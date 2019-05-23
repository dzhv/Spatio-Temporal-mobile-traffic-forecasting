import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
from os import path
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_folder)
import numpy as np

from experiment_builder import ExperimentBuilder
from models.lstm import LSTM

from data_providers.seq2seq_data_provider import Seq2SeqDataProvider
from data_providers.data_reader import FullDataReader
from data_providers.data_reader import MiniDataReader
import time

from arg_extractor import get_args

args = get_args() 
rng = np.random.RandomState(args.seed)

data_reader = MiniDataReader if args.use_mini_data else FullDataReader

experiment_builder = ExperimentBuilder(
	args = args,
	model = LSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size,
		learning_rate=args.learning_rate),
	experiment_name = args.experiment_name,
	num_epochs = args.num_epochs,
	train_data = Seq2SeqDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='train'), 
			window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
			shuffle_order=args.shuffle_order, rng=rng, fraction_of_data=args.fraction_of_data),
	val_data = Seq2SeqDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='valid'), 
			window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
			shuffle_order=args.shuffle_order, rng=rng),
)

experiment_builder.run_experiment()