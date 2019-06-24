python experiments/model_evaluator.py \
        --model_file results/predrnn_pred12/saved_models/train_model_13 \
        --model_name predrnn --batch_size 1 \
		--output_size 30 --hiden_size 30 --num_layers 2 \
		--shuffle_order false --prediction_batch_size 1  --evaluation_steps "10,12,30"
