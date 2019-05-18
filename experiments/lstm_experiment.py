import sys
from os import path
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_folder)
import numpy as np

from experiment_builder import ExperimentBuilder
from models.losses import nrmse_keras as nrmse
from models.lstm import LSTM

from data_providers.windowed_data_provider import WindowedDataProvider
from data_providers.data_reader import FullDataReader
from data_providers.data_reader import MiniDataReader
import time

from arg_extractor import get_args

DEFAULT_SEED = 12013094
rng = np.random.RandomState(DEFAULT_SEED)

batch_size = 1000
window_size = 11
segment_size = 12
hidden_size = 100

args = get_args() 

experiment_builder = ExperimentBuilder(
	model = LSTM(batch_size=batch_size, segment_size=segment_size, 
		num_features=window_size**2, hidden_size=hidden_size), 
	loss = nrmse, 
	experiment_name = "lstm",
	num_epochs = args.num_epochs,
	train_data = WindowedDataProvider(data_reader = MiniDataReader(data_folder=args.data_path, which_set='train'), 
			window_size=window_size, segment_size=segment_size, batch_size=batch_size,
			shuffle_order=True, rng=rng),
	val_data = WindowedDataProvider(data_reader = MiniDataReader(data_folder=args.data_path, which_set='valid'), 
			window_size=window_size, segment_size=segment_size, batch_size=batch_size,
			shuffle_order=True, rng=rng),
)

experiment_builder.run_experiment()