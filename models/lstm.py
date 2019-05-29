import sys
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, CuDNNLSTM, LSTM as CpuLSTM
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.test import is_gpu_available

from models.keras_model import KerasModel
from models import model_device_adapter

class LSTM(KerasModel):

	def __init__(self, gpus=0, batch_size=100, segment_size=12, num_features=121, 
		num_layers=2, hidden_size=100, learning_rate=0.0001, output_dim=1, create_tensorboard=False):
		lstm_cell = CuDNNLSTM if is_gpu_available() else CpuLSTM

		with tf.device('/cpu:0'):
			self.model = Sequential()
			input_shape = (segment_size, num_features)

			for i in range(num_layers - 1):
				self.model.add(lstm_cell(hidden_size, input_shape=input_shape, return_sequences=True))
			self.model.add(lstm_cell(hidden_size, input_shape=input_shape))

			self.model.add(Dense(output_dim))

		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)

		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(LSTM, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
