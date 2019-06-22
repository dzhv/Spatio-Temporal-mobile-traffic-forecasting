from experiment_builder import ExperimentBuilder
from models.cnn_convlstm import CnnConvLSTM
from models.keras_seq2seq import KerasSeq2Seq
from models.lstm import LSTM
from models.windowed_cnn_convlstm import WindowedCnnConvLSTM
from models.cnn_convlstm_seq2seq import CnnConvLSTMSeq2Seq
from models.cnn_convlstm_attention import CnnConvLSTMAttention
from models.predrnn.predrnn import PredRNN

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
	elif args.model_name == "cnn_convlstm_seq2seq":
		return CnnConvLSTMSeq2Seq(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			output_size=args.output_size, window_size=args.window_size, learning_rate=args.learning_rate, 
			learning_rate_decay=args.learning_rate_decay, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "cnn_convlstm_attention":
		return CnnConvLSTMAttention(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			window_size=args.window_size, learning_rate=args.learning_rate, 
			learning_rate_decay=args.learning_rate_decay, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "predrnn":
		return PredRNN(batch_size=args.batch_size, segment_size=args.segment_size, output_size=args.output_size,
			window_size=args.grid_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
			learning_rate=args.learning_rate)
	else:
		raise ValueError(f"Unknown model: {args.model_name}")
