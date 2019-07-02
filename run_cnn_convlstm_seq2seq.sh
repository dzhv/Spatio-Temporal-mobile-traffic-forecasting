#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=General_Usage
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=3-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate msc
python experiments/experiment_runner.py --data_path /home/${STUDENT_ID}/msc_project/data \
	  --experiment_name "cnn_convlstm_seq2seq_win11_1avg_decoder_valid" --model_name "cnn_convlstm_seq2seq" \
	  --gpus 1 --batch_size 1000 \
	  --cnn_filters "25,50,50" --encoder_filters "50,50,50" --decoder_filters "50,50,50" --mlp_hidden_sizes "50,1" \
	  --decoder_padding "valid" \
	  --learning_rate 0.0012 --learning_rate_decay 0.000005 --window_size 11 \
	  --num_epochs 150 --fraction_of_data 0.25
	  # !! decoder_padding valid !