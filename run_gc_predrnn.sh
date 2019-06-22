python experiments/experiment_runner.py \
	  --experiment_name "predrnn" --model_name "predrnn" \
	  --gpus 1 --batch_size 2500 \
	  --learning_rate 0.0013 --learning_rate_decay 0.000005 \
	  --num_epochs 150 \
	  --output_size 1 --hidden_size 100 --num_layers 2