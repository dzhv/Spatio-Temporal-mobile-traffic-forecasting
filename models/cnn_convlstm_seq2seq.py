import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.layers import Input, ConvLSTM2D, RNN, Dense, Conv2D, TimeDistributed, Conv2DTranspose, Flatten
from keras.layers import AveragePooling2D, UpSampling2D
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from models.keras_model import KerasModel
from models import model_device_adapter

class CnnConvLSTMSeq2Seq(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12, window_size=11,
		learning_rate=0.0001, create_tensorboard=False):

		self.segment_size = segment_size
		self.gpus = gpus
		
		# Define an input sequence.
		# 1 refers to a single channel of the input
		encoder_inputs = Input(shape=(segment_size, window_size, window_size, 1))
		
		out = TimeDistributed(Conv2D(32, kernel_size=3, activation='tanh', padding='same'))(encoder_inputs)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(64, kernel_size=3, activation='tanh', padding='same'))(out)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(64, kernel_size=3, activation='tanh', padding='same'))(out)

		# encoder
		encoder_outputs_and_states = ConvLSTM2D(filters=64, kernel_size=3, activation='tanh', 
			padding='same', return_state=True)(out)
		encoder_states = encoder_outputs_and_states[1:]

		# decoder
		
		latent_dim = window_size // 2 // 2  # accounting for average pooling operations
		self.decoder_input_shape = (latent_dim, latent_dim, 50)
		decoder_inputs = Input(shape=(segment_size,) + self.decoder_input_shape)
		out = ConvLSTM2D(filters=64, kernel_size=3, return_sequences=True, activation='tanh', 
			padding='same')(decoder_inputs, initial_state=encoder_states)

		out = TimeDistributed(Flatten())(out)

		num_output_features = 1
		  # TODO: this gets a 2400x1 (12x2x2x50) vector, maybe it's worth reducing the dimensions in lstm layers?
		out = TimeDistributed(Dense(num_output_features, activation='linear'))(out)

		self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=out)

		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(CnnConvLSTMSeq2Seq, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# adding an empty (channel) dimension to the end
		encoder_input = np.expand_dims(x, axis=-1)
		# (batch_size, segment_size, latent_dim, latent_dim, channels)
		decoder_input = np.zeros((encoder_input.shape[0], self.segment_size) + self.decoder_input_shape)
		
		return [encoder_input, decoder_input]

	def form_targets(self, y):
		return y[:, :, None]

model = CnnConvLSTMSeq2Seq(window_size=11)
output = model.forward(np.random.randn(1, 12, 11, 11))
print("output shape:")
print(output.shape)