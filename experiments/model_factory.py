from experiment_builder import ExperimentBuilder
from models.cnn_convlstm import CnnConvLSTM
from models.keras_seq2seq import KerasSeq2Seq
from models.lstm import LSTM
from models.windowed_cnn_convlstm import WindowedCnnConvLSTM
from models.cnn_convlstm_seq2seq import CnnConvLSTMSeq2Seq
from models.cnn_convlstm_attention import CnnConvLSTMAttention
from models.predrnn.predrnn import PredRNN
from models.predrnn.predrnn_windowed import PredRnnWindowed
from models.windowed_convlstm_seq2seq import WindowedConvLSTMSeq2Seq
from models.convlstm_seq2seq import ConvLSTMSeq2Seq
from models.mlp import MLP
from models.cnn_lstm import CnnLSTM

def get_model(args):
	if args.model_name == "lstm":
		return LSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size, 
			num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size,
			learning_rate=args.learning_rate, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "keras_seq2seq":
		return KerasSeq2Seq(batch_size=args.batch_size, segment_size=args.segment_size, 
			num_features=args.window_size**2, num_layers=args.num_layers, hidden_size=args.hidden_size,
			learning_rate=args.learning_rate, dropout=args.dropout, gpus=args.gpus, 
			output_size=args.output_size, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "cnn_convlstm":
		return CnnConvLSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			grid_size=args.grid_size, learning_rate=args.learning_rate, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "windowed_cnn_convlstm":
		return WindowedCnnConvLSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			window_size=args.window_size, learning_rate=args.learning_rate, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "cnn_convlstm_seq2seq":
		return CnnConvLSTMSeq2Seq(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			output_size=args.output_size, window_size=args.window_size, learning_rate=args.learning_rate, 
			cnn_filters=args.cnn_filters, encoder_filters=args.encoder_filters, decoder_filters=args.decoder_filters,
			mlp_hidden_sizes=args.mlp_hidden_sizes, decoder_padding=args.decoder_padding,
			learning_rate_decay=args.learning_rate_decay, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "cnn_convlstm_attention":
		return CnnConvLSTMAttention(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			window_size=args.window_size, learning_rate=args.learning_rate, output_size=args.output_size,
			cnn_filters=args.cnn_filters, encoder_filters=args.encoder_filters, decoder_filters=args.decoder_filters,
			pass_state=args.pass_state,
			learning_rate_decay=args.learning_rate_decay, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "convlstm_seq2seq":
		return ConvLSTMSeq2Seq(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			grid_size=args.grid_size, learning_rate=args.learning_rate, dropout=args.dropout,
			encoder_filters=args.encoder_filters, decoder_filters=args.decoder_filters, 
			kernel_size=args.kernel_size, output_size=args.output_size,
			learning_rate_decay=args.learning_rate_decay, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "windowed_convlstm_seq2seq":
		return WindowedConvLSTMSeq2Seq(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size,
			window_size=args.window_size, learning_rate=args.learning_rate, 
			encoder_filters=args.encoder_filters, decoder_filters=args.decoder_filters,
			learning_rate_decay=args.learning_rate_decay, create_tensorboard=args.create_tensorboard)
	elif args.model_name == "predrnn":
		return PredRNN(batch_size=args.batch_size, segment_size=args.segment_size, output_size=args.output_size,
			window_size=args.grid_size, hidden_sizes=args.hidden_sizes, learning_rate=args.learning_rate, 
			dropout=args.dropout)
	elif args.model_name == "windowed_predrnn":
		return PredRnnWindowed(batch_size=args.batch_size, segment_size=args.segment_size, 
			output_size=args.output_size, window_size=args.window_size, hidden_sizes=args.hidden_sizes, 
			mlp_hidden_sizes=args.mlp_hidden_sizes, learning_rate=args.learning_rate, 
			learning_rate_decay=args.learning_rate_decay)
	elif args.model_name == "mlp":
		return MLP(batch_size=args.batch_size, segment_size=args.segment_size, 
			window_size=args.window_size, hidden_sizes=args.hidden_sizes, 
			learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay)
	elif args.model_name == "cnn_lstm":
		return CnnLSTM(gpus=args.gpus, batch_size=args.batch_size, segment_size=args.segment_size, 
			output_size=args.output_size, window_size=args.window_size,
			cnn_filters=args.cnn_filters, hidden_sizes=args.hidden_sizes,
			learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay, 
			create_tensorboard=args.create_tensorboard)
	else:
		raise ValueError(f"Unknown model: {args.model_name}")
