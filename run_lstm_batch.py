import os

"""
experiment_name=$1
num_layers=$2
lr=$3
hidden_size=$4
"""
os.system("sbatch run_lstm.sh test 5 0.003 69")