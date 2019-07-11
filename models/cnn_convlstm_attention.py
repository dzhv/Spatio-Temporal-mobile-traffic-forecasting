import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.layers import Input, ConvLSTM2D, RNN, Dense, Conv2D, TimeDistributed, Conv2DTranspose, Flatten
from keras.layers import AveragePooling2D, UpSampling2D
from keras.layers.convolutional_recurrent import ConvRNN2D
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras import regularizers
import keras.backend as K
import numpy as np
import tensorflow as tf

from models.keras_model import KerasModel
from models import model_device_adapter
from models.convlstm_attention_cell import ConvLSTMAttentionCell

class CnnConvLSTMAttention(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12, output_size=12, window_size=11,
		cnn_filters=[25,50,50], encoder_filters=[50], decoder_filters=[50], pass_state=True,
		learning_rate=0.0001, learning_rate_decay=0, create_tensorboard=False):

		self.segment_size = segment_size
		self.output_size = output_size
		self.gpus = gpus
		
		# Define an input sequence.
		# 1 refers to a single channel of the input
		encoder_inputs = Input(shape=(segment_size, window_size, window_size, 1))
		
		print(f"encoder_inputs: {encoder_inputs}")
		out = TimeDistributed(Conv2D(cnn_filters[0], kernel_size=3, activation='tanh', 
			padding='same'))(encoder_inputs)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(cnn_filters[1], kernel_size=3, activation='tanh', padding='same'))(out)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(cnn_filters[2], kernel_size=3, activation='tanh', padding='same'))(out)

		# encoder
		for filters in encoder_filters[:-1]:
			out = ConvLSTM2D(filters=filters, kernel_size=3, return_sequences=True, activation='tanh', 
				padding='same')(out)

		encoder_outputs, state_h, state_c = ConvLSTM2D(filters=encoder_filters[-1], kernel_size=3, 
			activation='tanh', padding='same', return_state=True, return_sequences=True)(out)
		# encoder_outputs shape: (batch_size, segment_size, window_size, window_size, num_filters)

		# decoder

		latent_dim = window_size // 2**(len(cnn_filters) - 1)
		self.decoder_input_shape = (latent_dim, latent_dim, encoder_filters[-1])
		decoder_inputs = Input(shape=(output_size,) + self.decoder_input_shape, name="decoder_input")
	
		attention_layer = ConvRNN2D(ConvLSTMAttentionCell(decoder_filters[0], kernel_size=3, padding='same'), return_sequences=True)
		attention_layer._num_constants = 1
		# the second encoder_outputs is given as 'constants' to the layer,
		# so that it can be fully transferred to the cell
		if pass_state:
			out = attention_layer([decoder_inputs, state_h, state_c, encoder_outputs])
		else:
			out = attention_layer([decoder_inputs, encoder_outputs])

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

		super(CnnConvLSTMAttention, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# adding an empty (channel) dimension to the end
		encoder_input = np.expand_dims(x, axis=-1)

		# (batch_size, output_size, latent_dim, latent_dim, channels)
		decoder_input = np.zeros((encoder_input.shape[0], self.output_size) + self.decoder_input_shape)
		
		return [encoder_input, decoder_input]

	def form_targets(self, y):
		return y[:, :, None]

if __name__ == '__main__':
	model = CnnConvLSTMAttention(window_size=11, output_size=3)
	output = model.forward(np.random.randn(2, 12, 11, 11))
	print("output shape:")
	print(output.shape)