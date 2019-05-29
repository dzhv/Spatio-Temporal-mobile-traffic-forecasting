import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from data_providers.full_grid_data_provider import FullGridDataProvider
from data_providers.data_reader import MiniDataReader
from mock_data_reader import MockDataReader

import numpy as np

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
	 [2, 2, 7]],

 	[[4, 0, 2],
	 [5, 1, 0],
	 [2, 1, 2]],
])

segment_size=1
target_segment_size=3
batch_size=2
data_reader = MockDataReader(data)
sut = FullGridDataProvider(data_reader, segment_size=segment_size, target_segment_size=target_segment_size,
        batch_size=batch_size, shuffle_order=False)	

def shape_test():
	expected_inputs_shape = (batch_size, segment_size, data.shape[-2], data.shape[-1])
	expected_targets_shape = (batch_size, target_segment_size, data.shape[-2], data.shape[-1])

	for inputs, targets in sut.next():
		print("-----")
		assert inputs.shape == expected_inputs_shape, \
			f"inputs had the wrong shape. Expected: {expected_inputs_shape}, got: {inputs.shape}"

		assert targets.shape == expected_targets_shape, \
			f"outputs had the wrong shape. Expected: {expected_targets_shape}, got: {targets.shape}"

		print("inputs:")
		print(inputs)
		print("targets:")
		print(targets)

shape_test()


print("TEST PASSED")

	