import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from experiment_builder import ExperimentBuilder
from models.losses import nrmse_numpy as nrmse
from models.mean_predictor import MeanPredictor

from data_providers.mini_data_provider import MiniDataProvider
from data_providers.data_reader import SingleFileReader

rng = np.random.RandomState(DEFAULT_SEED)

experiment_builder = ExperimentBuilder(
	model = MeanPredictor(mean=57.277), 
	loss = nrmse, 
	experiment_name = "mean_predictor",
	num_epochs = 2, 
	train_data = MiniDataProvider(data_reader = SingleFileReader('train'), 
			window_size=11, segment_size=12,
            batch_size=10, segment_batch_size=30, shuffle_order=True, rng=rng),
	val_data = MiniDataProvider(data_reader = SingleFileReader('valid'), 
			window_size=11, segment_size=12,
            batch_size=10, segment_batch_size=30, shuffle_order=True, rng=rng),
    continue_from_epoch=-1)

experiment_builder.run_experiment()