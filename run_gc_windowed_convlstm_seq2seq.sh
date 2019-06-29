python experiments/experiment_runner.py \
	  --experiment_name "convlstm_seq2seq_mini_windowed" --model_name "windowed_convlstm_seq2seq" \
	  --gpus 1 --batch_size 2500 \
	  --learning_rate 0.0025 --learning_rate_decay 0.000005 \
	  --num_epochs 150 \
	  --output_size 12 --encoder_filters "32,64,64" --decoder_filters "128, 64, 32, 5" \
	  --use_mini_data true