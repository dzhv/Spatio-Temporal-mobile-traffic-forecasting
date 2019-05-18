class MockDataReader(object):
	def __init__(self, mock_data):
		self.mock_data = mock_data

	def next(self):
		return self.mock_data

class MockIterativeDataReader(object):
	def __init(self, mock_data):
		# mock_data - should be an iteratable collection of data
		self.mock_data = mock_data

	def next(self):
		for data in mock_data:
			yield data

