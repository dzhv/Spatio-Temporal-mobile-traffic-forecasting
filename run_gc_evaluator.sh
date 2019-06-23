python experiments/model_evaluator.py \
        --model_file results/keras_seq2seq_2lay_00003lr_100hs/saved_models/train_model_latest \
        --model_name keras_seq2seq --batch_size 1000 --num_layers 2 \
        --hiden_size 100 \
        --shuffle_order false --prediction_batch_size 10000  --evaluation_steps "10,12"