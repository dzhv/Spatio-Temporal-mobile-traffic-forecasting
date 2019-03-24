import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np

import data_grid_mapping

def assertions(result):
	assert result[0, 99, 0] == 2
	assert result[1, 99, 0] == 3
	assert result[2, 99, 0] == 4

	assert result[0, 99, 0] == 2
	assert result[1, 99, 0] == 3
	assert result[2, 99, 0] == 4

	assert result[0, 1, 98] == 2
	assert result[1, 1, 98] == 0
	assert result[2, 1, 98] == 4

test_file = "raw_input.txt"

def test_map():	
	result = data_grid_mapping.map(data_path=test_file)
	assertions(result)

	print("test_map successful")

def test_map_and_save():
	save_path = "test.npy"
	data_grid_mapping.map_and_save(test_file, save_path)

	result = np.load(save_path)
	assertions(result)

	print("test_map_and_save successful")

test_map()
test_map_and_save()