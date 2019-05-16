class Model(object):
	def reset_parameters(self):
		pass

	def use_gpu(self, gpu):
		pass

	def train_mode(self):  # activate training mode
		pass

	def eval_mode(self): # activate evaluation mode
		pass

	def forward(self, x):  # compute model output with x inputs
		pass

	def loss(self, predictions, targets):
		pass