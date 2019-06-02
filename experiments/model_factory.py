from experiment_builder import ExperimentBuilder
from models.cnn_convlstm import CnnConvLSTM
from models.keras_seq2seq import KerasSeq2Seq
from models.lstm import LSTM
from models.windowed_cnn_convlstm import WindowedCnnConvLSTM

def get_model(args):
	if args.model_name == "lstm":
		return LSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size, 
		num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size,
		learning_rate=args.learning_rate, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "keras_seq2seq":
		return KerasSeq2Seq(batch_size=args.batch_size, segment_size=args.segment_size, 
			num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size,
			learning_rate=args.learning_rate, dropout=args.dropout, gpus=args.gpus, 
			create_tensorboard=args.create_tensorboard)
	elif args.model_name == "cnn_convlstm":
		return CnnConvLSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			grid_size=args.grid_size, learning_rate=args.learning_rate, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "windowed_cnn_convlstm":
		return WindowedCnnConvLSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			window_size=args.window_size, learning_rate=args.learning_rate, create_tensorboard=args.create_tensorboard)
	else:
		raise ValueError(f"Unknown model: {args.model_name}")
