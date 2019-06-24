python experiments/experiment_runner.py \
	  --experiment_name "predrnn_2" --model_name "predrnn" \
	  --gpus 1 --batch_size 1 \
	  --learning_rate 0.0013 --learning_rate_decay 0.000005 \
	  --num_epochs 150 \
	  --output_size 12 --hidden_sizes "128,64,64,64"