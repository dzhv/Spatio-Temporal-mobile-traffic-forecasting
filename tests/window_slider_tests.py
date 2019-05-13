import numpy as np 
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from data_providers import window_slider

def test(data, expected_windows1, expected_windows2, window_size):
	actual = window_slider.get_windowed_data(data, window_size)

	for window in expected_windows1:		
		assert (actual[0] == window).all((1, 2)).any(), f"window:\n{window}\nwas not found in the output"

	assert len(actual[1]) == 9

	for window in expected_windows2:
		assert (actual[1] == window).all((1, 2)).any(), f"window:\n{window}\nwas not found in the output"		

	print("test passed")

# 2 x 4 x 4
data = np.array([
	[[2, 3, 4],
	 [3, 4, 5],
	 [4, 5, 6]],

	[[3, 2, 1],
	 [0, 3, 4],
	 [6, 5, 4]]
	])

window_size = 3

expected_windows1 = np.array([
	[[0, 0, 0],
	 [0, 2, 3],
	 [0, 3, 4]],

	[[0, 0, 0],
	 [2, 3, 4],
	 [3, 4, 5]],

	 [[0, 0, 0],
	 [3, 4, 0],
	 [4, 5, 0]],

	[[0, 2, 3],
	 [0, 3, 4],
	 [0, 4, 5]],

	 [[2, 3, 4],
	 [3, 4, 5],
	 [4, 5, 6]],

	[[3, 4, 0],
	 [4, 5, 0],
	 [5, 6, 0]],

	 [[0, 3, 4],
	 [0, 4, 5],
	 [0, 0, 0]],

	[[3, 4, 5],
	 [4, 5, 6],
	 [0, 0, 0]],

	 [[4, 5, 0],
	 [5, 6, 0],
	 [0, 0, 0]]
 ])

expected_windows2 = np.array([
	[[3, 2, 1],
	 [0, 3, 4],
	 [6, 5, 4]],

	[[2, 1, 0],
	 [3, 4, 0],
	 [5, 4, 0]]
])

test(data, expected_windows1, expected_windows2, window_size)

