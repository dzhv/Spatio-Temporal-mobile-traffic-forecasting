python experiments/experiment_runner.py
	  --experiment_name "cnn_convlstm_seq2seq_win11_pred30_mini" --model_name "cnn_convlstm_seq2seq" \
	  --gpus 1 --batch_size 1000 \
	  --cnn_filters "25,50,50" --encoder_filters "50,50" --decoder_filters "50,50" --mlp_hidden_sizes "50,1" \
	  --output_size 30 --learning_rate 0.0025 --learning_rate_decay 0.000005 --window_size 11 \
	  --num_epochs 150 --use_mini_data true
	  # !! careful, output size == 30 !!