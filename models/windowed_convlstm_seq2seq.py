import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.layers import Input, ConvLSTM2D, RNN, Dense, Conv2D, TimeDistributed, Conv2DTranspose, Flatten
from keras.layers import AveragePooling2D, UpSampling2D
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import tensorflow as tf

from models.keras_model import KerasModel
from models import model_device_adapter

class WindowedConvLSTMSeq2Seq(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12, output_size=12, window_size=11,
		encoder_filters=[50], decoder_filters=[50],
		learning_rate=0.0001, learning_rate_decay=0, create_tensorboard=False):

		self.segment_size = segment_size
		self.output_size = output_size
		self.gpus = gpus
		
		# Define an input sequence.
		# 1 refers to a single channel of the input
		encoder_inputs = Input(shape=(segment_size, window_size, window_size, 1))		

		out = encoder_inputs

		# encoder

		for filters in encoder_filters[:-1]:
			out = ConvLSTM2D(filters=filters, kernel_size=3, return_sequences=True, activation='tanh', 
				padding='same')(out)

		encoder_outputs, state_h, state_c = ConvLSTM2D(filters=encoder_filters[-1], kernel_size=3, 
			activation='tanh', padding='same', return_state=True, return_sequences=True)(out)

		# decoder
		
		self.decoder_input_shape = (window_size, window_size, encoder_filters[-1])
		decoder_inputs = Input(shape=(output_size,) + self.decoder_input_shape)

		# first decoder layer gets the encoder states
		out = ConvLSTM2D(filters=decoder_filters[0], kernel_size=3, return_sequences=True, activation='tanh', 
			padding='same')([decoder_inputs, state_h, state_c])

		for filters in decoder_filters[1:]:
			out = ConvLSTM2D(filters=filters, kernel_size=3, return_sequences=True, activation='tanh', 
				padding='same')(out)

		out = TimeDistributed(Flatten())(out)

		num_output_features = 1
		out = TimeDistributed(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.002)))(out)
		out = TimeDistributed(Dense(num_output_features, activation='linear'))(out)

		self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=out)

		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate, decay=learning_rate_decay)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(WindowedConvLSTMSeq2Seq, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# adding an empty (channel) dimension to the end
		encoder_input = np.expand_dims(x, axis=-1)
		# (batch_size, segment_size, latent_dim, latent_dim, channels)
		decoder_input = np.zeros((encoder_input.shape[0], self.output_size) + self.decoder_input_shape)
		
		return [encoder_input, decoder_input]

	def form_targets(self, y):
		return y[:, :, None]

if __name__ == '__main__':
	model = WindowedConvLSTMSeq2Seq(window_size=11)
	output = model.forward(np.random.randn(2, 12, 11, 11))
	print("output shape:")
	print(output.shape)