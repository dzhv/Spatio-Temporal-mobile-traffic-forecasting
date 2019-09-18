python experiments/experiment_runner.py \
	  --experiment_name "cnn_convlstm_attention_mini" --model_name "cnn_convlstm_attention" \
	  --gpus 1 --batch_size 2500 \
	  --cnn_filters "25,50,50" --encoder_filters "50,50" --decoder_filters "50,50" \
	  --learning_rate 0.0025 --learning_rate_decay 0.00002 --window_size 11 \
	  --num_epochs 150 --fraction_of_data 0.25 \
	  --use_mini_data true
