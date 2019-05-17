import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as np

from experiment_builder import ExperimentBuilder
from models.losses import nrmse
from models.lstm import LSTM

from data_providers.mini_data_provider import MiniDataProvider
from data_providers.data_reader import SingleFileReader

DEFAULT_SEED = 12013094
rng = np.random.RandomState(DEFAULT_SEED)

batch_size = 10
window_size = 11
segment_size = 12
segment_batch_size = 30
hidden_size = 100

experiment_builder = ExperimentBuilder(
	model = LSTM(batch_size=batch_size, segment_size=segment_size, 
		num_features=window_size**2, hidden_size=hidden_size), 
	loss = nrmse, 
	experiment_name = "lstm",
	num_epochs = 5,
	train_data = MiniDataProvider(data_reader = SingleFileReader('train'), 
			window_size=window_size, segment_size=segment_size, batch_size=batch_size,
			segment_batch_size=segment_batch_size, shuffle_order=True, rng=rng),
	val_data = MiniDataProvider(data_reader = SingleFileReader('valid'), 
			window_size=window_size, segment_size=segment_size, batch_size=batch_size,
			segment_batch_size=segment_batch_size, shuffle_order=True, rng=rng),
)

experiment_builder.run_experiment()