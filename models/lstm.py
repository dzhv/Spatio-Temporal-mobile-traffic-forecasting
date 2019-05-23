from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, CuDNNLSTM
from tensorflow.keras.layers import LSTM as CpuLSTM
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.test import is_gpu_available
from models.model import Model
import tensorflow as tf

import sys

class LSTM(Model):

	def __init__(self, gpus=0, batch_size=100, segment_size=12, num_features=121, 
		num_layers=2, hidden_size=100, learning_rate=0.0001, output_dim=1):
		self.batch_size = batch_size		

		lstm_cell = CuDNNLSTM if is_gpu_available() else CpuLSTM

		with tf.device('/cpu:0'):
			self.model = Sequential()
			input_shape = (segment_size, num_features)

			for i in range(num_layers - 1):
				self.model.add(lstm_cell(hidden_size, input_shape=input_shape, return_sequences=True))
			self.model.add(lstm_cell(hidden_size, input_shape=input_shape))

			self.model.add(Dense(output_dim))

		if is_gpu_available():
			try:
				self.model = multi_gpu_model(self.model, gpus=gpus)
				print("\nUsing multiple gpus\n")
			except:
				print("\nUsing single GPU\n")
		else:
			print("\nUsing CPU LSTM!\n")

		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss=mean_squared_error, optimizer=optimizer)

		print(self.model.summary())

	def reshape_inputs(self, x):
		return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

	def forward(self, x):
		x_reshaped = self.reshape_inputs(x)
		return self.model.predict(x_reshaped, batch_size=self.batch_size)

	def train(self, x, y):
		""" inputs:
				x - (batch_size, segment_size, window_width, window_height)
				y - (batch_size,)
		"""		

		x_reshaped = self.reshape_inputs(x)
		history = self.model.fit(x_reshaped, y, batch_size=self.batch_size, epochs=1)
		# print(history.history)
		return history.history["loss"][0]

	def evaluate(self, x, y):
		x_reshaped = self.reshape_inputs(x)
		return self.model.evaluate(x_reshaped, y, batch_size=y.shape[0])

	def save(self, path):
		self.model.save(path + ".h5")

	def load(self, path):
		self.model = load_model(path)
