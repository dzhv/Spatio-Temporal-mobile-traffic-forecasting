import os

"""
experiment_name=$1
num_layers=$2
lr=$3
hidden_size=$4
"""

layers = [1, 2, 3]
learning_rates = [0.001, 0.0001, 0.00005]
hidden_sizes = [50]#, 75, 100]


for n_layers in layers:
	for lr in learning_rates:
		for hidden_size in hidden_sizes:
			experiment_name = "lstm_{0}layer_{1}lr_{2}hs".format(n_layers, str(lr).replace('.', ''), hidden_size)
			print("experiment name: {0}".format(experiment_name))

			command = "sbatch run_lstm.sh {0} {1} {2} {3}".format(
				experiment_name, n_layers, lr, hidden_size)
			print("command: {0}".format(command))

			os.system(command)


