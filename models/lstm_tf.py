import tensorflow as tf
from tensorflow.contrib import rnn
from models.model import Model

class LSTM(Model):
	def __init__(self, gpus=0, batch_size=100, segment_size=12, num_features=121, 
		num_layers=2, hidden_size=100, learning_rate=0.001, model=None, session_holder=None):

		self.num_features = num_features
		self.hidden_size = hidden_size
		self.session_holder = session_holder
		self.segment_size = segment_size

		self.X = tf.placeholder("float", [None, segment_size, num_features])
		self.Y = tf.placeholder("float", [None, 1])

		# Define weights
		self.linear_weights = tf.Variable(tf.random_normal([self.hidden_size, 1]))
		
		self.linear_biases = tf.Variable(tf.random_normal([1]))

		self.forward_op = self.RNN(self.X, self.linear_weights, self.linear_biases)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.loss = tf.losses.mean_squared_error(self.Y, self.forward_op)
		self.train_op = self.optimizer.minimize(self.loss)

	def reshape_inputs(self, x):
		return x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

	def RNN(self, x, linear_weights, linear_biases):
		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, timesteps, num_features)
		# Required shape: 'timesteps' tensors list of shape (batch_size, num_features)

		# Unstack to get a list of 'num_features' tensors of shape (batch_size, num_features)
		x = tf.unstack(x, self.segment_size, 1)

		# Define a lstm cell with tensorflow
		lstm_cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)

		# Get lstm cell output
		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		# Linear activation, using rnn inner loop last output
		return tf.matmul(outputs[-1], linear_weights) + linear_biases

	def forward(self, x):
		x_reshaped = self.reshape_inputs(x)
		return self.session_holder.sess.run(self.forward_op, feed_dict={self.X: x_reshaped})

	def train(self, x, y):
		""" inputs:
				x - (batch_size, segment_size, window_width, window_height)
				y - (batch_size,)
		"""		

		x_reshaped = self.reshape_inputs(x)
		y = y[:, None]
		self.session_holder.sess.run(self.train_op, feed_dict={self.X: x_reshaped, self.Y: y})
		return self._evaluate(x_reshaped, y)

	def evaluate(self, x, y):
		y = y[:, None]
		x = self.reshape_inputs(x)
		return self._evaluate(x, y)

	def _evaluate(self, x, y):
		return self.session_holder.sess.run(self.loss, feed_dict={self.X: x, self.Y: y})

	def save(self, path):
		print("WARNING: this model does not have a saving function")

	def reset_parameters(self):
		self.session_holder.sess.run(tf.global_variables_initializer())

