import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.layers import Input, ConvLSTM2D, RNN, Dense, Conv2D, TimeDistributed, Conv2DTranspose, Flatten
from keras.layers import AveragePooling2D, UpSampling2D
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import numpy as np

from models.keras_model import KerasModel
from models import model_device_adapter

class CnnConvLSTMSeq2Seq(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12, window_size=15,
		learning_rate=0.0001, create_tensorboard=False):

		self.segment_size = segment_size

		# Define an input sequence.
		# 1 refers to a single channel of the input
		encoder_inputs = Input(shape=(segment_size, window_size, window_size, 1))
		
		out = TimeDistributed(Conv2D(25, kernel_size=3, activation='tanh', padding='same'))(encoder_inputs)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(50, kernel_size=3, activation='tanh', padding='same'))(out)
		out = TimeDistributed(AveragePooling2D())(out)
		out = TimeDistributed(Conv2D(50, kernel_size=3, activation='tanh', padding='same'))(out)

		# encoder
		out = ConvLSTM2D(filters=50, kernel_size=3, return_sequences=True, activation='tanh', padding='same')(out)
		encoder_outputs, state_h, state_c = ConvLSTM2D(filters=50, kernel_size=3, activation='tanh', 
			padding='same', return_state=True)(out)
		
		# decoder
		decoder_inputs = Input(shape=(segment_size, 3, 3, 50))  # here (3, 3) is the latent dimension - not kernel size
		out = ConvLSTM2D(filters=50, kernel_size=3, return_sequences=True, activation='tanh', 
			padding='same')(decoder_inputs, initial_state=[state_h, state_c])
		out = ConvLSTM2D(filters=50, kernel_size=3, return_sequences=True, activation='tanh', padding='same')(out)

		out = Flatten()(out)

		num_output_features = 1
		  # TODO: this gets a 5400x1 (12x3x3x50) vector, maybe it's worth reducing the dimensions in lstm layers?
		decoder_dense = Dense(num_output_features, activation='linear')
		decoder_outputs = decoder_dense(out)

		self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=out)
		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(CnnConvLSTMSeq2Seq, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def form_model_inputs(self, x):
		# (batch_size, segment_size, grid_size, grid_size)
		# print(x.shape)

		# adding an empty (channel) dimension to the end
		return np.expand_dims(x, axis=-1)

	def form_model_inputs(self, x):
		# adding an empty (channel) dimension to the end
		encoder_input = np.expand_dims(x, axis=-1)
		# (batch_size, segment_size, latent_dim, latent_dim, channels)
		decoder_input = np.zeros((encoder_input.shape[0], self.segment_size, 3, 3, 50))
		return [encoder_input, decoder_input]

	def form_targets(self, y):
		return y[:, :, None]

# model = CnnConvLSTMSeq2Seq()
# output = model.forward(np.random.randn(1, 12, 15, 15))
# print("output shape:")
# print(output.shape)