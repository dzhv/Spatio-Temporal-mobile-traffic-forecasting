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

class CnnConvLSTMSeq2Seq(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12, output_size=12, window_size=11,
		cnn_filters=[2,3,4], encoder_filters=[5,6], decoder_filters=[6,7], mlp_hidden_sizes=[50, 1],
		decoder_padding='same', learning_rate=0.0001, learning_rate_decay=0, create_tensorboard=False):

		self.segment_size = segment_size
		self.output_size = output_size
		self.gpus = gpus
		
		# Define an input sequence.
		# 1 refers to a single channel of the input
		encoder_inputs = Input(shape=(segment_size, window_size, window_size, 1), name="encoder_input")

		# cnns
		
		out = TimeDistributed(Conv2D(cnn_filters[0], kernel_size=3, activation='tanh', padding='same'), 
			name="cnn_1")(encoder_inputs)
		for i, filters in enumerate(cnn_filters[1:]):
			out = TimeDistributed(AveragePooling2D(), name=f"avg_pool_{i + 1}")(out)
			out = TimeDistributed(Conv2D(filters, kernel_size=3, activation='tanh', padding='same'),
				name=f"cnn_{i+2}")(out)

		# encoder
		for i, filters in enumerate(encoder_filters[:-1]):
			out = ConvLSTM2D(filters=filters, kernel_size=3, return_sequences=True, activation='tanh', 
				padding='same', name=f"encoder_convlstm_{i+1}")(out)

		encoder_outputs, state_h, state_c = ConvLSTM2D(filters=encoder_filters[-1], kernel_size=3, 
			activation='tanh', padding='same', return_state=True, return_sequences=True,
			name=f"encoder_convlstm_{len(encoder_filters)}")(out)

		# decoder

		latent_dim = window_size // 2**(len(cnn_filters) - 1)
		self.decoder_input_shape = (latent_dim, latent_dim, encoder_filters[-1])
		decoder_inputs = Input(shape=(output_size,) + self.decoder_input_shape, name="decoder_input")
		
		out = ConvLSTM2D(filters=decoder_filters[0], kernel_size=3, return_sequences=True, activation='tanh', 
			padding='same', name="decoder_convlstm_1")([decoder_inputs, state_h, state_c])

		for i, filters in enumerate(decoder_filters[1:]):
			out = ConvLSTM2D(filters=filters, kernel_size=3, return_sequences=True, activation='tanh', 
				padding=decoder_padding, name=f"decoder_convlstm_{i+2}")(out)

		# predictor mlp

		out = TimeDistributed(Flatten(), name="flatten")(out)

		for i, hidden_size in enumerate(mlp_hidden_sizes[:-1]):
			out = TimeDistributed(Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(0.002)), 
				name=f"mlp_{i}")(out)

		out = TimeDistributed(Dense(mlp_hidden_sizes[-1], activation='linear', 
			kernel_regularizer=regularizers.l2(0.002)), name="mlp_final")(out)


		self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=out)

		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate, decay=learning_rate_decay)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(CnnConvLSTMSeq2Seq, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# adding an empty (channel) dimension to the end
		encoder_input = np.expand_dims(x, axis=-1)
		# (batch_size, segment_size, latent_dim, latent_dim, channels)
		decoder_input = np.zeros((encoder_input.shape[0], self.output_size) + self.decoder_input_shape)
		
		return [encoder_input, decoder_input]

	def form_targets(self, y):
		return y[:, :, None]

if __name__ == '__main__':
	model = CnnConvLSTMSeq2Seq(window_size=11, decoder_padding='valid', 
		cnn_filters=[1,2], encoder_filters=[3,4], decoder_filters=[4,5,6])
	output = model.forward(np.random.randn(2, 12, 11, 11))
	print("output shape:")
	print(output.shape)	

	model.train(np.random.randn(2, 12, 11, 11), np.random.randn(2, 12))
	print("train success")

	# model.save('temp')

	# model = CnnConvLSTMSeq2Seq(window_size=11, output_size=30)
	# model.load('temp')

	# out = model.forward(np.random.randn(2, 12, 11, 11))
	# print(f"out shape: {out.shape}")
	
