import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from data_providers.mini_data_provider import MiniDataProvider
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
	 [2, 2, 7]]
])

window_size = 3
segment_size = 2
data_reader = MockDataReader(data)
batch_size = 3
sut = MiniDataProvider(data_reader, window_size=window_size, segment_size=segment_size,
        batch_size=batch_size, segment_chunk_size=4, shuffle_order=False)	

def shape_test():
	for inputs, outputs in sut.next():
		print("-----")
		assert inputs.shape == (batch_size, segment_size, window_size, window_size), \
			f"inputs had the wrong shape. Expected: {batch_shape}, got: {inputs.shape}"

		assert outputs.shape == (batch_size,), \
			f"outputs had the wrong shape. Expected: {(batch_size,)}, got: {outputs.shape}"

		print(inputs)
		print(outputs)

	print("TEST PASSED")

def number_of_samples_test():
	n_samples = len(list(sut.next())) * batch_size

	expected = 18

	assert n_samples == expected, f"expected {expected} samples, got: {n_samples}"

	print("TEST PASSED")

number_of_samples_test()




	