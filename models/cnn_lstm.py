import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from keras.layers import Input, RNN, Dense, Conv2D, TimeDistributed, LSTMCell, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import tensorflow as tf

from models.keras_model import KerasModel
from models import model_device_adapter

class CnnLSTM(KerasModel):
	def __init__(self, gpus=1, batch_size=50, segment_size=12, output_size=12, window_size=15,
		cnn_filters=[2,3,4], hidden_sizes=[10,10],
		learning_rate=0.0001, learning_rate_decay=0, create_tensorboard=False):

		self.segment_size = segment_size
		self.output_size = output_size
		self.gpus = gpus
		
		# Define an input sequence.
		# 1 refers to a single channel of the input
		inputs = Input(shape=(segment_size, window_size, window_size, 1), name="input")

		# cnns
		
		out = TimeDistributed(Conv2D(cnn_filters[0], kernel_size=5, activation='relu', padding='same'), 
			name="cnn_1")(inputs)
		out = TimeDistributed(MaxPooling2D(), name=f"max_pool")(out)
		out = TimeDistributed(Conv2D(cnn_filters[1], kernel_size=5, activation='relu', padding='same'), 
			name="cnn_2")(out)
		out = TimeDistributed(AveragePooling2D(), name=f"avg_pool_1")(out)
		out = TimeDistributed(Conv2D(cnn_filters[2], kernel_size=5, activation='relu', padding='same'), 
			name="cnn_3")(out)
		out = TimeDistributed(AveragePooling2D(), name=f"avg_pool_2")(out)

		out = TimeDistributed(Flatten(), name="flatten_before_lstm")(out)

		cells = [LSTMCell(hidden_sizes[0]), LSTMCell(hidden_sizes[1])]
		out = RNN(cells)(out)

		# out = Flatten(name="flatten_after_lstm")(out)

		out = Dense(100, activation='relu', name=f"mlp_relu")(out)		
		out = Dense(output_size, activation='linear', name=f"mlp_linear")(out)


		self.model = Model(inputs=inputs, outputs=out)
		self.model = model_device_adapter.get_device_specific_model(self.model, gpus)
		
		optimizer = Adam(lr=learning_rate, decay=learning_rate_decay)
		self.model.compile(loss='mse', optimizer=optimizer)

		print(self.model.summary())

		super(CnnLSTM, self).__init__(batch_size=batch_size, create_tensorboard=create_tensorboard)


	# TODO: overwrite train, evaluate, forward

	def form_model_inputs(self, x):
		# adding an empty (channel) dimension to the end
		input = np.expand_dims(x, axis=-1)
		
		return input

if __name__ == '__main__':
	model = CnnLSTM(window_size=11,
		cnn_filters=[1,2,3], hidden_sizes=[10,12])
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
	
