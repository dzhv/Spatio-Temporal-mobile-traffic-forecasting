class Model(object):
	def reset_parameters(self):
		pass

	def use_gpu(self, gpu):
		pass

	def train_mode(self):  # activate training mode
		pass

	def eval_mode(self): # activate evaluation mode
		pass

	def train(self, x, y):  # train using the batch (x, y)
		return 0	# return loss

	def forward(self, x):  # compute model output with x inputs
		pass

	def loss(self, predictions, targets):
		pass

