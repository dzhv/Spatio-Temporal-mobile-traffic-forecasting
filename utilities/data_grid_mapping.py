import numpy as np
import pandas as pd

import milano_grid

from datetime import datetime
import os

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

def map_december():
	print("Mapping all December files")

	input_folder = "data/december_input"
	output_folder = "data/december_mapped"
	map_month(input_folder, output_folder, "2013-12", 31)

def map_november():
	print("Mapping all November files")

	input_folder = "data/november_input"
	output_folder = "data/november_mapped"
	map_month(input_folder, output_folder, "2013-11", 30, start_from=8)

def map_month(input_folder, output_folder, year_month_string, number_of_days, start_from = 1):	
	os.system(f"mkdir {output_folder}")
	file_name_template = f"sms-call-internet-mi-{year_month_string}-"

	for i in range(start_from, number_of_days + 1):
		two_digit_i = "{:02d}".format(i)
		input_file = f"{input_folder}/{file_name_template}{two_digit_i}.txt"
		output_file = f"{output_folder}/{year_month_string}-{two_digit_i}.npy"
		map_and_save(input_file, output_file)

# map_december()
# map_november()
map_and_save(data_path="data/december_input/sms-call-internet-mi-2014-01-01.txt",
	save_path="data/2014-01-01.npy")
# TODO: map november
# TODO: map 2014-01-01

