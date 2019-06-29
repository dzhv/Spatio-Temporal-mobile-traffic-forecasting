python experiments/model_evaluator.py \
        --model_file results/cnn_convlstm_seq2seq_win11_3/saved_models/train_model_40 \
        --model_name cnn_convlstm_attention --batch_size 1000 \
		--cnn_filters="25,50,50" --encoder_filters="50,50,50" --decoder_filters="50,50,50" \
		--mlp_hidden_sizes="1" --output_size 30 \
		--shuffle_order false --prediction_batch_size 10000  --evaluation_steps "10,12,30"
