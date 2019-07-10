import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

from models.keras_model import KerasModel
from models import model_device_adapter

class MLP(KerasModel):

	def __init__(self, gpus=0, batch_size=100, segment_size=12, window_size=11, 
		hidden_sizes=[10,12], learning_rate=0.0001, learning_rate_decay=0, create_tensorboard=False):
		num_features = segment_size * window_size ** 2

		self.model = Sequential()

		self.model.add(Dense(hidden_sizes[0], input_shape=(num_features,), activation='relu'))
		
		for hidden_size in hidden_sizes[1:-1]:
			self.model.add(Dense(hidden_size, activation='relu'))

		self.model.add(Dense(hidden_sizes[-1]))

		# self.model = model_device_adapter.get_device_specific_model(self.model, gpus)

		optimizer = Adam(lr=learning_rate, decay=learning_rate_decay)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(MLP, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# x.shape == (batch_size, segment_size, window_size, window_size)
		return x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

if __name__ == '__main__':
	model = MLP()
	output = model.forward(np.random.randn(2, 12, 11, 11))
	print("output shape:")
	print(output.shape)	

	model.train(np.random.randn(2, 12, 11, 11), np.random.randn(2, 12))
	print("train success")

