from .model import Model
import numpy as np
from .losses import mse

class MeanPredictor(Model):
	def __init__(self, mean):
		self.mean = mean

	def forward(self, x):  # compute model output with x inputs
		return np.zeros(x.shape[0]) + self.mean

	def train(self, x, y):
		return mse(y, self.mean)

	def evaluate(self, x, y):
		return mse(y, self.mean)

	def save(self, path):
		pass