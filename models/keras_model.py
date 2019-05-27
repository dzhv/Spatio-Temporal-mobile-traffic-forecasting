from models.model import Model
import abc
class KerasModel(Model):
	# A wrapper class for Keras Models
	# containing common functionality

	__metaclass__ = abc.ABCMeta

	def __init__(self, batch_size, create_tensorboard=False, tensorboard_dir='logs/tensorboard'):
		self.batch_size = batch_size
		self.create_tensorboard = create_tensorboard
		self.tensorboard_dir = tensorboard_dir

	@abc.abstractmethod
	def form_model_inputs(self, x):
		"""	abstract method where each subclass implements
			the shape of the model inputs
		"""

	def form_targets(self, y):
		""" method which can be potentially overriden in subclasses
			if some adjustmenst to targets are needed
		"""
		return y

	def forward(self, x):
		inputs = self.form_model_inputs(x)
		return self.model.predict(inputs, batch_size=self.batch_size)

	def train(self, x, y):
		""" inputs:
				x - (batch_size, segment_size, window_width, window_height)
				y - (batch_size,)
		"""		

		inputs = self.form_model_inputs(x)
		targets = self.form_targets(y)

		callbacks = []
		if self.create_tensorboard:
			callbacks.append(TensorBoard(log_dir=self.tensorboard_dir, 
			write_grads=True, write_graph=False, write_images=True))		
		
		history = self.model.fit(inputs, targets, batch_size=self.batch_size, epochs=1,
			callbacks=callbacks)

		# self.step_num += 1

		# gradients = K.gradients(self.model.output, self.model.input)  
		# sess = K.get_session()
		# evaluated_gradients = sess.run(gradients[0], feed_dict={self.model.input: x_reshaped})
		# print("\nHere be gradients:")
		# print(evaluated_gradients)
		# print("")

		assert len(history.history["loss"]) == 1, \
			f"training history contained something more than 1 loss: {history.history}"
		return history.history["loss"][0]

	def evaluate(self, x, y):
		inputs = self.form_model_inputs(x)
		targets = self.form_targets(y)
		return self.model.evaluate(inputs, targets, batch_size=targets.shape[0])

	def save(self, path):
		self.model.save_weights(path + ".h5")

	def load(self, path):
		full_path = path + ".h5"
		print(f"Loading model from {full_path}\n")
		self.model.load_weights(full_path)

	
