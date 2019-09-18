python experiments/experiment_runner.py \
	--experiment_name "windowed_predrnn_mini" --model_name "windowed_predrnn" \
	--batch_size 2500 --use_mini_data true \
	--learning_rate 0.001 --learning_rate_decay 0.000001 \
	--segment_size 12 --output_size 12 \
	--hidden_sizes "128,64,64,64" --mlp_hidden_sizes "100,1"
	
