class SessionHolder(object):
	def __init__(self, sess=None):
		self.sess = sess

	def set_sess(self, sess):
		self.sess = sess