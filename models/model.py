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

	def evaluate(self, x, y): # compute predictions and loss w.r.t. targets
		return 0	# return loss

	def loss(self, predictions, targets):
		pass

	def save(self, path): # save the model to a file, path is without extension
		pass

