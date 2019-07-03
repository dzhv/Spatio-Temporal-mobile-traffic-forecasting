import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.layers import Input, ConvLSTM2D, RNN, Dense, Conv2D, TimeDistributed, Conv2DTranspose, Flatten
from keras.layers import AveragePooling2D, UpSampling2D, Dropout
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import tensorflow as tf

from models.keras_model import KerasModel
from models import model_device_adapter

import keras.backend as K

class ConvLSTMSeq2Seq(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12, output_size=12, grid_size=100,
		encoder_filters=[50], decoder_filters=[50,1], dropout=0, kernel_size=3,
		learning_rate=0.0001, learning_rate_decay=0, create_tensorboard=False):

		print(f"!!!kernel_size: {kernel_size}")

		self.segment_size = segment_size
		self.output_size = output_size
		self.gpus = gpus
		
		# Define an input sequence.
		# 1 refers to a single channel of the input
		encoder_inputs = Input(shape=(segment_size, grid_size, grid_size, 1))		

		out = encoder_inputs

		# encoder

		for i, filters in enumerate(encoder_filters[:-1]):
			out = Dropout(dropout)(out)
			out = ConvLSTM2D(filters=filters, kernel_size=kernel_size, return_sequences=True, activation='tanh', 
				padding='same', name=f"encoder_{i+1}")(out)

		out = Dropout(dropout)(out)
		encoder_outputs, state_h, state_c = ConvLSTM2D(filters=encoder_filters[-1], kernel_size=kernel_size, 
			activation='tanh', padding='same', return_state=True, return_sequences=True, 
			name=f"encoder_{len(encoder_filters)}")(out)

		# decoder
		
		self.decoder_input_shape = (grid_size, grid_size, encoder_filters[-1])
		decoder_inputs = Input(shape=(output_size,) + self.decoder_input_shape, name="decoder_input")

		# first decoder layer gets the encoder states
		out = ConvLSTM2D(filters=decoder_filters[0], kernel_size=kernel_size, return_sequences=True, 
			activation='tanh', padding='same', name="decoder_1")([decoder_inputs, state_h, state_c])

		for i, filters in enumerate(decoder_filters[1:]):
			out = Dropout(dropout)(out)
			out = ConvLSTM2D(filters=filters, kernel_size=kernel_size, return_sequences=True, activation='tanh', 
				padding='same', name=f"decoder_{i+2}")(out)

		# cnn forming the final outputs, so that predictions can be outside the range of tanh activation
		out = TimeDistributed(Conv2D(1, kernel_size=1, activation='linear', padding='same'), 
			name=f"cnn_final")(out)

		self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=out)

		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate, decay=learning_rate_decay)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(ConvLSTMSeq2Seq, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# adding an empty (channel) dimension to the end
		encoder_input = np.expand_dims(x, axis=-1)
		# (batch_size, segment_size, latent_dim, latent_dim, channels)
		decoder_input = np.zeros((encoder_input.shape[0], self.output_size) + self.decoder_input_shape)
		
		return [encoder_input, decoder_input]

	def form_targets(self, y):
		# adding an empty (channel) dimension to the end
		return np.expand_dims(y, axis=-1)

if __name__ == '__main__':
	model = ConvLSTMSeq2Seq(grid_size=10)
	output = model.forward(np.random.randn(2, 12, 10, 10))
	print("output shape:")
	print(output.shape)

	model.train(np.random.randn(2, 12, 10, 10), np.random.randn(2, 12, 10, 10))
	print("train success")




