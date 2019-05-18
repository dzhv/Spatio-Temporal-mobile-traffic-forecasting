import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from data_providers.windowed_data_provider import WindowedDataProvider
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
	 [2, 2, 7]]
])

window_size = 3
segment_size = 2
data_reader = MockDataReader(data)
batch_size = 3
sut = WindowedDataProvider(data_reader, window_size=window_size, segment_size=segment_size,
        batch_size=batch_size, shuffle_order=False)	

def shape_test():
	for inputs, outputs in sut.next():
		print("-----")
		assert inputs.shape == (batch_size, segment_size, window_size, window_size), \
			f"inputs had the wrong shape. Expected: {batch_shape}, got: {inputs.shape}"

		assert outputs.shape == (batch_size,), \
			f"outputs had the wrong shape. Expected: {(batch_size,)}, got: {outputs.shape}"

		print(inputs)
		print(outputs)

def number_of_samples_test():
	n_samples = len(list(sut.next())) * batch_size
	expected = 18

	assert n_samples == expected, f"expected {expected} samples, got: {n_samples}"

def test():
	dr = MiniDataReader()
	sut = WindowedDataProvider(dr, window_size=11, segment_size=12,
            batch_size=1000, shuffle_order=False)	

	for x, y in sut.next():
		print(f"{x.shape} , {y.shape}")

shape_test()
number_of_samples_test()

print("TEST PASSED")

	