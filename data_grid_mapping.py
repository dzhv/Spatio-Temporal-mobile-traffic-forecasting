import numpy as np
import pandas as pd

import milano_grid

from datetime import datetime

def map_to_tensor(data, grid_size):
	# returns a 3D tensor of size (T, grid_height, grid_length)
	# where T is the time axis

	groups = data.groupby("time")

	tensor = np.empty((0, grid_size[0], grid_size[1]))
	i = 0
	n_groups = len(groups)
	for time, group in groups:
		print(timestamp_string(time))
		grid = np.zeros(grid_size)

		for index, item in group.iterrows():
			grid_loc = milano_grid.map(item.square_id)			
			grid[grid_loc] += item.internet

		# add empty first dimension, so that grid dimensions would match tensor's
		grid_3d = np.expand_dims(grid, axis=0)
		tensor = np.append(tensor, grid_3d, axis=0)

		i += 1
		print("{0}/{1} groups processed".format(i, n_groups))
	
	return tensor

def get_sorted_data(data_path):
	print("reading the data from file")
	data = pd.read_csv(data_path, delimiter="\t", header=None, 
		usecols=[0, 1, 7], names=["square_id", "time", "internet"])

	cleaned_data = data.dropna()
	return cleaned_data.sort_values(by="time")

def map(data_path):
	sorted_data = get_sorted_data(data_path)
	tensor = map_to_tensor(sorted_data, milano_grid.size)

	print(tensor.shape)
	return tensor

def map_and_save(data_path, save_path):
	print("Mapping the data")
	tensor = map(data_path=data_path)

	print("Saving the data")
	np.save(save_path, tensor)

def timestamp_string(timestamp):
	ts_in_seconds = int(timestamp) // 1000
	return datetime.utcfromtimestamp(ts_in_seconds).strftime('%Y-%m-%d %H:%M:%S')

def print_times(data_path):
	sorted_data = get_sorted_data(data_path)

	previous = None
	indx = 0
	for timestamp in sorted_data["time"]:
		if timestamp != previous:			
			previous = timestamp
				
			print("{0}, indx: {1}".format(timestamp_string(timestamp), indx))
			indx += 1

data_path = "data/sms-call-internet-mi-2013-11-01.txt" # "data/mini_data.dat" 
save_path = "data/11-01_mapped.npy"

# map(data_path)

