from .model import Model
import numpy as np

class MeanPredictor(Model):
	def __init__(self, mean):
		self.mean = mean

	def forward(self, x):  # compute model output with x inputs
		return np.zeros(x.shape[0]) + self.mean

	def loss(self, predictions, targets):
		pass