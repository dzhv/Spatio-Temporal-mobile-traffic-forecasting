import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.layers import Input, ConvLSTM2D, RNN, Dense, Conv2D, TimeDistributed, Conv2DTranspose, Flatten
from keras.layers import AveragePooling2D, UpSampling2D
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import numpy as np

from models.keras_model import KerasModel
from models import model_device_adapter

class WindowedCnnConvLSTM(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12,
		window_size=15, learning_rate=0.0001, create_tensorboard=False):

		self.segment_size = segment_size

		# Define an input sequence.
		# 1 refers to a single channel of the input
		inputs = Input(shape=(segment_size, window_size, window_size, 1))
		
		out = TimeDistributed(Conv2D(25, kernel_size=3, activation='tanh', padding='same'))(inputs)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(50, kernel_size=3, activation='tanh', padding='same'))(out)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(50, kernel_size=3, activation='tanh', padding='same'))(out)

		out = ConvLSTM2D(filters=50, kernel_size=3, return_sequences=True, activation='tanh', padding='same')(out)
		out = ConvLSTM2D(filters=50, kernel_size=3, return_sequences=True, activation='tanh', padding='same')(out)
		out = ConvLSTM2D(filters=25, kernel_size=3, activation='tanh', padding='same')(out)

		out = Flatten()(out)
		out = Dense(1)(out)

		# out = Conv2DTranspose(50, kernel_size=3, activation='tanh', padding='same')(out)		
		# out = UpSampling2D()(out)
		# out = Conv2DTranspose(50, kernel_size=3, activation='tanh', padding='same')(out)
		# out = UpSampling2D()(out)
		# out = Conv2DTranspose(25, kernel_size=3, activation='tanh', padding='same')(out)
		# out = UpSampling2D()(out)
		# out = Conv2DTranspose(10, kernel_size=3, activation='tanh')(out)		
		# out = Conv2DTranspose(1, kernel_size=3, activation='tanh')(out)

		self.model = Model(inputs=inputs, outputs=out)
		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(WindowedCnnConvLSTM, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# (batch_size, segment_size, grid_size, grid_size)
		# print(x.shape)

		# adding an empty (channel) dimension to the end
		return np.expand_dims(x, axis=-1)

# model = WindowedCnnConvLSTM()
# output = model.forward(np.random.randn(1, 12, 15, 15))
# print("output shape:")
# print(output.shape)