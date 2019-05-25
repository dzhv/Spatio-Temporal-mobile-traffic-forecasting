import seq2seq
from seq2seq.models import Seq2Seq
from models.model import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Sequential
from keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.test import is_gpu_available

class LstmSeq2Seq(Model):
	def __init__(self, gpus=0, batch_size=100, segment_size=12, num_features=121, 
		num_layers=2, hidden_size=10, learning_rate=0.0001, dropout=0, model=None,
		create_tensorboard=False):

		self.batch_size = batch_size

		if model is not None:
			self.model = model
			return
		
		with tf.device('/cpu:0'):
			self.model = Seq2Seq(batch_input_shape=(batch_size, segment_size, num_features), 
				hidden_dim=hidden_size, output_length=segment_size, output_dim=1, depth=num_layers, dropout=dropout)

		if is_gpu_available():
			try:
				self.model = multi_gpu_model(self.model, gpus=gpus)
				print("\nUsing multiple gpus\n")
			except:
				print("\nUsing single GPU\n")
		else:
			print("\nUsing CPU\n")

		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)		

		self.create_tensorboard = create_tensorboard
		self.step_num = 0

		print(self.model.summary())

	def reshape_inputs(self, x):
		return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

	def forward(self, x):
		x_reshaped = self.reshape_inputs(x)
		return self.model.predict(x_reshaped, batch_size=self.batch_size)

	def train(self, x, y):
		""" inputs:
				x - (batch_size, segment_size, window_width, window_height)
				y - (batch_size,)
		"""		

		x_reshaped = self.reshape_inputs(x)
		y = y[:, :, None]

		callbacks = []
		if self.create_tensorboard:
			callbacks.append(TensorBoard(log_dir='logs/seq2seq_2', 
			histogram_freq=1, write_grads=True, write_graph=False, write_images=True))

		history = self.model.fit(x_reshaped, y, batch_size=self.batch_size, epochs=self.step_num + 1,
			callbacks=callbacks, initial_epoch=self.step_num)

		self.step_num += 1

		# gradients = K.gradients(self.model.output, self.model.input)  
		# sess = K.get_session()
		# evaluated_gradients = sess.run(gradients[0], feed_dict={self.model.input: x_reshaped})
		# print("\nHere be gradients:")
		# print(evaluated_gradients)
		# print("")

		return history.history["loss"][0]

	def evaluate(self, x, y):
		x_reshaped = self.reshape_inputs(x)
		return self.model.evaluate(x_reshaped, y, batch_size=y.shape[0])

	def save(self, path):
		self.model.save_weights(path + ".h5")

	def load(self, path):
		self.model.load_weights(path)
