import numpy as np

def combine_train():
	path = "data/train/"
	num_files = 42

	data = None
	for i in range(1, num_files+1):
		month_day = i if i <= 30 else i - 30
		two_digit_day = "{:02d}".format(month_day)
		file = f"2013-11-{two_digit_day}.npy" if i <= 30 else f"2013-12-{two_digit_day}.npy"
		file = path + file
		print(f"Reading file: {file}")

		file_data = np.load(file)

		data = file_data if data is None else np.concatenate((data, file_data))

	expected_shape = (144 * num_files, 100, 100)
	assert data.shape == expected_shape, f"Expected data shape of {expected_shape}, got: {data.shape}"

	np.save("data/train_raw.npy", data)


def combine_val():
	path = "data/val/"
	num_files = 10
	start_from = 13

	data = None
	for i in range(start_from, start_from + num_files):		
		two_digit_day = "{:02d}".format(i)
		file = path + f"2013-12-{two_digit_day}.npy"
		print(f"Reading file: {file}")

		file_data = np.load(file)

		data = file_data if data is None else np.concatenate((data, file_data))

	expected_shape = (144 * num_files, 100, 100)
	assert data.shape == expected_shape, f"Expected data shape of {expected_shape}, got: {data.shape}"

	np.save("data/val_raw.npy", data)

def combine_test():
	path = "data/test/"
	num_files = 10
	start_from = 23

	data = None
	for i in range(start_from, start_from + num_files):		
		month_day = i if i <= 31 else i - 31
		two_digit_day = "{:02d}".format(month_day)
		file = f"2013-12-{two_digit_day}.npy" if i <= 31 else f"2014-01-{two_digit_day}.npy"
		file = path + file
		print(f"Reading file: {file}")

		file_data = np.load(file)

		data = file_data if data is None else np.concatenate((data, file_data))

	expected_shape = (144 * num_files, 100, 100)
	assert data.shape == expected_shape, f"Expected data shape of {expected_shape}, got: {data.shape}"

	np.save("data/test_raw.npy", data)

combine_val()