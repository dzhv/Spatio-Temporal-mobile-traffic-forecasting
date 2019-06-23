#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Short
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-04:00:00

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

python experiments/model_evaluator.py --data_path /home/${STUDENT_ID}/msc_project/data \
		  --model_file /results/keras_seq2seq_2lay_00003lr_100hs/saved_models/train_model_latest \
		  --model_name keras_seq2seq --batch_size 1000 --num_layers 2 \
		  --hiden_size 100 \
		  --shuffle_order false --prediction_batch_size 10000  --evaluation_steps "10,12"