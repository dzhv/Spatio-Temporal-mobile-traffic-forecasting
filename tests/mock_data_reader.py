class MockDataReader(object):
	def __init__(self, mock_data):
		self.mock_data = mock_data

	def next(self):
		return self.mock_data