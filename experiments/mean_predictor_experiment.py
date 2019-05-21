import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from experiment_builder import ExperimentBuilder
from models.losses import nrmse_numpy as nrmse
from models.mean_predictor import MeanPredictor

from data_providers.windowed_data_provider import WindowedDataProvider
from data_providers.data_reader import FullDataReader, MiniDataReader
import numpy as np
from arg_extractor import get_args

args = get_args() 

rng = np.random.RandomState(args.seed)

experiment_builder = ExperimentBuilder(
	args = args,
	model = MeanPredictor(mean=0),
	experiment_name = "mean_predictor",
	num_epochs = 1,
	train_data = WindowedDataProvider(data_reader=FullDataReader(args.data_path, 'train'), 
			window_size=11, segment_size=12,
            batch_size=args.batch_size, shuffle_order=True, rng=rng),
	val_data = WindowedDataProvider(data_reader=FullDataReader(args.data_path, 'valid'), 
			window_size=11, segment_size=12,
            batch_size=args.batch_size, shuffle_order=True, rng=rng),
    continue_from_epoch=-1)

experiment_builder.run_experiment()