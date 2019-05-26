# Code adapted from: https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction

from keras.layers import Input, LSTMCell, RNN, Dense
from models.model import Model
from keras.models import Model as KerasModel
from keras.optimizers import Adam
import numpy as np

class KerasSeq2Seq(Model):
	def __init__(self, gpus=0, batch_size=100, segment_size=12, num_features=121, 
		num_layers=2, hidden_size=10, learning_rate=0.0001, dropout=0, create_tensorboard=False):

		self.batch_size = batch_size
		self.segment_size = segment_size

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
		# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
		num_output_features = 1
		decoder_dense = Dense(num_output_features, activation='linear')  # TODO: try regularizers

		decoder_outputs = decoder_dense(decoder_outputs)

		self.model = KerasModel(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
		
		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())


	def create_stacked_lstms(self, hidden_size, num_layers, return_sequences, return_state):
		# Create a list of RNN Cells, these are then concatenated into a single layer
		# with the RNN layer.
		cells = []
		for i in range(num_layers):
		    cells.append(LSTMCell(hidden_size)) # TODO: try regularizers

		return RNN(cells, return_sequences=return_sequences, return_state=return_state)

	def reshape_inputs(self, x):
		return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

	def forward(self, x):
		x_reshaped = self.reshape_inputs(x)
		decoder_input = np.zeros((x_reshaped.shape[0], self.segment_size, 1))
		return self.model.predict([x_reshaped, decoder_input], batch_size=self.batch_size)

	def train(self, x, y):
		""" inputs:
				x - (batch_size, segment_size, window_width, window_height)
				y - (batch_size,)
		"""		

		x_reshaped = self.reshape_inputs(x)
		y = y[:, :, None]

		callbacks = []
		# if self.create_tensorboard:
		# 	callbacks.append(TensorBoard(log_dir='logs/seq2seq_2', 
		# 	write_grads=True, write_graph=False, write_images=True))

		# (batch_size, segment_length, 1)
		# print(y.shape)

		# (input batch size, segment_size, number of features FOR each prediction time step)
		decoder_input = np.zeros((y.shape[0], self.segment_size, 1))
		history = self.model.fit([x_reshaped, decoder_input], y, batch_size=self.batch_size, epochs=1,
			callbacks=callbacks)

		# self.step_num += 1

		# gradients = K.gradients(self.model.output, self.model.input)  
		# sess = K.get_session()
		# evaluated_gradients = sess.run(gradients[0], feed_dict={self.model.input: x_reshaped})
		# print("\nHere be gradients:")
		# print(evaluated_gradients)
		# print("")

		print(history.history)
		return history.history["loss"][0]

	def evaluate(self, x, y):
		x_reshaped = self.reshape_inputs(x)
		decoder_input = np.zeros((y.shape[0], self.segment_size, 1))
		return self.model.evaluate([x_reshaped, decoder_input], y, batch_size=y.shape[0])

	def save(self, path):
		self.model.save_weights(path + ".h5")

	def load(self, path):
		self.model.load_weights(path + ".h5")