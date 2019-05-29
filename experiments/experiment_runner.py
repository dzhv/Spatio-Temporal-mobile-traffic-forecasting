import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
from os import path
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_folder)
import numpy as np

from experiment_builder import ExperimentBuilder
from models.cnn_convlstm import CnnConvLSTM
from data_providers.full_grid_data_provider import FullGridDataProvider
from data_providers import data_provider_factory
from arg_extractor import get_args
import model_factory

args = get_args() 
rng = np.random.RandomState(args.seed)

model = model_factory.get_model(args)
train_data, val_data = data_provider_factory.get_data_providers(args, rng)

experiment_builder = ExperimentBuilder(
	args = args,
	model = model,
	train_data = train_data, 
	val_data = val_data,
	experiment_name = args.experiment_name,
	num_epochs = args.num_epochs,	
	continue_from_epoch=args.continue_from_epoch
)

experiment_builder.run_experiment()