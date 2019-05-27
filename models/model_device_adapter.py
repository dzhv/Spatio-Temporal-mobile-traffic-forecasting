from tensorflow.keras.utils import multi_gpu_model
from tensorflow.test import is_gpu_available

def get_device_specific_model(model, gpus):
	""" utility function for preparing Keras models to use specific available devices
	"""
	if is_gpu_available():
		try:
			model = multi_gpu_model(model, gpus)
			print("\nUsing multiple gpus\n")
		except:
			print("\nUsing single GPU\n")
	else:
		print("\nUsing CPU!\n")

	return model