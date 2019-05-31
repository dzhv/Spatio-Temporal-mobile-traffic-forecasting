from keras.utils import multi_gpu_model
from tensorflow.test import is_gpu_available

def get_device_specific_model(model, gpus):
	""" utility function for preparing Keras models to use specific available devices
	"""
	print(f"is gpu available: {is_gpu_available()}")
	print(f"gpus: {gpus}")
	if is_gpu_available():
		try:
			print(f"trying to use {gpus} gpus")
			model = multi_gpu_model(model, gpus)
			print("\nUsing multiple gpus\n")
		except Exception as inst:
			print(type(inst))
			print(inst.args)
			print(inst)

			print("\nUsing single GPU\n")
	else:
		print("\nUsing CPU!\n")

	return model