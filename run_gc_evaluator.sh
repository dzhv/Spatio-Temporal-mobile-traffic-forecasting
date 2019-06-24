python experiments/model_evaluator.py \
        --model_file results/cnn_convlstm_attention/saved_models/train_model_latest \
        --model_name cnn_convlstm_attention --batch_size 1000 \
		--cnn_filters="20,40,40" --decoder_filters="10,64" --encoder_filters="40" \
		--shuffle_order false --prediction_batch_size 10000  --evaluation_steps "10,12"
