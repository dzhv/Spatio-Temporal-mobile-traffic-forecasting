# Code adapted from: https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction

from keras.layers import Input, LSTMCell, RNN, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np

from models.keras_model import KerasModel
from models import model_device_adapter


class KerasSeq2Seq(KerasModel):
	def __init__(self, gpus=1, batch_size=100, segment_size=12, num_features=121,
		num_layers=2, hidden_size=10, learning_rate=0.0001, dropout=0, 
		output_size=12, create_tensorboard=False):

		self.segment_size = segment_size
		self.output_size = output_size

		# create encoder and decoder LSTM towers/stacks
		encoder = self.create_stacked_lstms(hidden_size=hidden_size, num_layers=num_layers,
			return_sequences=False, return_state=True)
		decoder = self.create_stacked_lstms(hidden_size=hidden_size, num_layers=num_layers,
			return_sequences=True, return_state=True)

		# Define an input sequence.
		encoder_inputs = Input(shape=(None, num_features))

		encoder_outputs_and_states = encoder(encoder_inputs)
		# Discard encoder outputs and only keep the states.
		# The outputs are of no interest to us, the encoder's
		# job is to create a state describing the input sequence.
		# encoder_states = [state_h, state_c]
		encoder_states = encoder_outputs_and_states[1:]


		# The decoder input will be set to zero as decoder only relies on the encoder state
		decoder_inputs = Input(shape=(None, 1))
		# Set the initial state of the decoder to be the ouput state of the encoder.
		# This is the fundamental part of the encoder-decoder.
		decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)
		# Only select the output of the decoder (not the states)
		decoder_outputs = decoder_outputs_and_states[0]

		
		# TODO: try with and without this dense layer

		# Apply a dense layer with linear activation to set output to correct dimension
		# and scale
		num_output_features = 1
		decoder_dense = Dense(num_output_features, activation='linear')  # TODO: try regularizers

		decoder_outputs = decoder_dense(decoder_outputs)

		self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(KerasSeq2Seq, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)

	def create_stacked_lstms(self, hidden_size, num_layers, return_sequences, return_state):
		# Create a list of RNN Cells, these are then concatenated into a single layer
		# with the RNN layer.
		cells = []
		for i in range(num_layers):
		    cells.append(LSTMCell(hidden_size)) # TODO: try regularizers

		return RNN(cells, return_sequences=return_sequences, return_state=return_state)

	def form_model_inputs(self, x):
		encoder_input = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
		# (batch_size, segment_size, arbitrary input dimension)
		decoder_input = np.zeros((encoder_input.shape[0], self.output_size, 1))
		return [encoder_input, decoder_input]

	def form_targets(self, y):
		return y[:, :, None]

if __name__ == '__main__':
	model = KerasSeq2Seq(output_size=6)
	output = model.forward(np.random.randn(2, 12, 11, 11))
	print("output shape:")
	print(output.shape)	

	model.train(np.random.randn(2, 12, 11, 11), np.random.randn(2, 6))
	print("train success")
