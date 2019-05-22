import seq2seq
from seq2seq.models import Seq2Seq
from models.model import Model
from keras.optimizers import Adam

class LstmSeq2Seq(Model):
	def __init__(self, gpus=0, batch_size=100, segment_size=12, num_features=121, 
		num_layers=2, hidden_size=10, depth=2, learning_rate=0.0001, model=None):

		self.batch_size = batch_size

		if model is not None:
			self.model = model
			return
		
		self.model = Seq2Seq(
			batch_input_shape=(batch_size, segment_size, num_features), 
			hidden_dim=hidden_size, output_length=segment_size, output_dim=1, depth=depth
		)
		optimizer = Adam(lr=learning_rate)
		self.model.compile(loss='mse', optimizer=optimizer)

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
		history = self.model.fit(x_reshaped, y, batch_size=self.batch_size, epochs=1)
		print(history.history)
		return history.history["loss"][0]

	def evaluate(self, x, y):
		x_reshaped = self.reshape_inputs(x)
		return self.model.evaluate(x_reshaped, y, batch_size=y.shape[0])

	def save(self, path):
		self.model.save_weights(path + ".h5")

	def load(self, path):
		self.model.load_weights(path)
