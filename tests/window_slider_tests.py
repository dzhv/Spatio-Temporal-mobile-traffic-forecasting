import numpy as np 
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from data_providers import window_slider

# def assert_window_exists(window, tensor):
# 	assert (tensor == window).all()

def get_windowed_data_test():
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


	actual = window_slider.get_windowed_data(data, window_size)

	for window in expected_windows1:		
		assert (actual[:, 0, :, :] == window).all((1, 2)).any(), \
			f"window:\n{window}\nwas not found in the output"

	assert len(actual[:, 1, :, :]) == 9

	for window in expected_windows2:
		assert (actual[:, 1, :, :] == window).all((1, 2)).any(), \
			f"window:\n{window}\nwas not found in the output"		

	print("test passed")

def find_segment(tensor, segment):
	search_flags = (tensor == segment).all((1, 2, 3))

	# assert it exists
	assert search_flags.any(), f"segment:\n{segment}\nwas not found in the output"

	# return the index of the segment
	return np.where(search_flags == True)[0]

def get_windowed_segmented_data_test():

	data = np.array([
		[[2, 3, 4],
		 [3, 4, 5],
		 [4, 5, 6]],

		[[3, 2, 1],
		 [0, 3, 4],
		 [6, 5, 4]],

		[[4, 1, 8],
		 [0, 2, 5],
		 [3, 3, 0]],

		[[9, 9, 6],
		 [1, 4, 3],
		 [2, 2, 7]]
	])

	window_size = 3
	segment_size = 2

	actual_inputs, actual_targets = window_slider.get_windowed_segmented_data(data, window_size, segment_size)

	expected_input_1 = np.array([
		[[2, 3, 4],
		 [3, 4, 5],
		 [4, 5, 6]],

		[[3, 2, 1],
		 [0, 3, 4],
		 [6, 5, 4]],		 
	])
	expected_target_1 = 2

	expected_input_2 = np.array([		# lower right corner
		[[3, 4, 0],
		 [5, 4, 0],
		 [0, 0, 0]],

		[[2, 5, 0],
		 [3, 0, 0],
		 [0, 0, 0]],		 
	])
	expected_target_2 = 7

	expected_input_3 = np.array([		# 2 last rows
		[[0, 3, 4],
		 [6, 5, 4],
		 [0, 0, 0]],

		[[0, 2, 5],
		 [3, 3, 0],
		 [0, 0, 0]],		 
	])
	expected_target_3 = 2

	test_cases = [(expected_input_1, expected_target_1), (expected_input_2, expected_target_2), 
		(expected_input_3, expected_target_3)]

	for expected_input, expected_target in test_cases:		
		index = find_segment(actual_inputs, expected_input)

		# assert the target is correct for the segment	
		assert actual_targets[index] == expected_target, \
			f"The target for segment:\n{expected_input}\n {actual_targets[index]}" + \
				f"does not match the expected: {expected_target}"

		print("test case passed")

	print("TEST SUCCESSFUL")

def get_sequential_inputs_and_targets_test():
	data = np.array([
		[[2, 3, 4],
		 [3, 4, 5],
		 [4, 5, 6]],

		[[3, 2, 1],
		 [0, 3, 4],
		 [6, 5, 4]],

		[[4, 1, 8],
		 [0, 2, 5],
		 [3, 3, 0]],

		[[9, 9, 6],
		 [1, 4, 3],
		 [2, 2, 7]]
	])

	inputs, targets = window_slider.get_sequential_inputs_and_targets(data, 3, 2)

	expected_input_shape = (9, 2, 3, 3)
	assert inputs.shape == expected_input_shape, \
		f"expected input shape to be {expected_input_shape}, was: {inputs.shape}"
	expected_targets_shape = (9, 2)
	assert targets.shape == expected_targets_shape, \
		f"expected targets shape to be {expected_targets_shape}, was: {targets.shape}"

		
	print(inputs)
	print("---------")
	print(targets)


get_sequential_inputs_and_targets_test()

