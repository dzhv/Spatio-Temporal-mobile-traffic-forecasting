from data_providers.data_reader import FullDataReader
from data_providers.data_reader import MiniDataReader
from data_providers.windowed_data_provider import WindowedDataProvider
from data_providers.full_grid_data_provider import FullGridDataProvider
from data_providers.seq2seq_data_provider import Seq2SeqDataProvider

def get_data_providers(args, rng, test_set=False):
	data_reader = MiniDataReader if args.use_mini_data else FullDataReader

	if args.model_name in ['lstm', 'windowed_cnn_convlstm']:
		return get_windowed_data_providers(args, rng, data_reader, test_set)
	elif args.model_name in ['keras_seq2seq', 'cnn_convlstm_seq2seq', 'cnn_convlstm_attention', 
		'windowed_predrnn', 'windowed_convlstm_seq2seq', 
		'mlp', 'cnn_lstm']:
		return get_seq2seq_data_providers(args, rng, data_reader, test_set)
	elif args.model_name in ['cnn_convlstm', 'predrnn', 'convlstm_seq2seq']:
		return get_full_grid_data_providers(args, rng, data_reader, test_set)
	else:
		raise ValueError(f"Unknown model: {args.model_name}")

def get_windowed_data_providers(args, rng, data_reader, test_set):
	if test_set:
		return WindowedDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='test'), 
			window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
			shuffle_order=args.shuffle_order, rng=rng, fraction_of_data=args.fraction_of_data)

	train_data = WindowedDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='train'), 
			window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
			shuffle_order=args.shuffle_order, rng=rng, fraction_of_data=args.fraction_of_data)
	val_data = WindowedDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='valid'), 
			window_size=args.window_size, segment_size=args.segment_size, batch_size=args.batch_size,
			shuffle_order=args.shuffle_order, rng=rng, fraction_of_data=args.fraction_of_val)

	return train_data, val_data

def get_seq2seq_data_providers(args, rng, data_reader, test_set):
	if test_set:
		return Seq2SeqDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='test'), 
			window_size=args.window_size, segment_size=args.segment_size, output_size=args.output_size,
			batch_size=args.batch_size, shuffle_order=args.shuffle_order, rng=rng, 
			fraction_of_data=args.fraction_of_data, missing_data=args.missing_data)

	train_data = Seq2SeqDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='train'), 
			window_size=args.window_size, segment_size=args.segment_size, output_size=args.output_size,
			batch_size=args.batch_size, shuffle_order=args.shuffle_order, rng=rng, 
			fraction_of_data=args.fraction_of_data)
	val_data = Seq2SeqDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='valid'), 
			window_size=args.window_size, segment_size=args.segment_size, output_size=args.output_size,
			batch_size=args.batch_size, shuffle_order=args.shuffle_order, rng=rng, 
			fraction_of_data=args.fraction_of_val)

	return train_data, val_data

def get_full_grid_data_providers(args, rng, data_reader, test_set):
	if test_set:
		return FullGridDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='test'), 
			segment_size=args.segment_size, batch_size=args.batch_size, target_segment_size=args.output_size,
			shuffle_order=args.shuffle_order, rng=rng, fraction_of_data=args.fraction_of_data, 
			missing_data=args.missing_data)

	train_data = FullGridDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='train'), 
			segment_size=args.segment_size, batch_size=args.batch_size, target_segment_size=args.output_size,
			shuffle_order=args.shuffle_order, rng=rng, fraction_of_data=args.fraction_of_data)
	val_data = FullGridDataProvider(data_reader = data_reader(data_folder=args.data_path, which_set='valid'), 
			segment_size=args.segment_size, batch_size=args.batch_size, target_segment_size=args.output_size,
			shuffle_order=args.shuffle_order, rng=rng, fraction_of_data=args.fraction_of_data)

	return train_data, val_data