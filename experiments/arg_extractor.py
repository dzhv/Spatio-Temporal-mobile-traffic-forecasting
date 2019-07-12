import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2int_list(v):
    try:
        return [int(i) for i in v.split(',')]
    except:
        raise argparse.ArgumentTypeError(f"Failed to parse: {v} into an integer list.")

def get_args():
    """
    Returns a named tuple with arguments extracted from the command line.
    :return: A named tuple with arguments
    """
    parser = argparse.ArgumentParser()

    # data shape params
    parser.add_argument('--batch_size', nargs="?", type=int, default=1000, help='Batch_size for experiment')
    parser.add_argument('--segment_size', nargs="?", type=int, default=12,
                        help='segment size of 1 training/validation sample')
    parser.add_argument('--output_size', nargs="?", type=int, default=12,
                        help='output (prediction) length')
    parser.add_argument('--window_size', nargs="?", type=int, default=11,
                        help='size of the segment elements')
    parser.add_argument('--grid_size', nargs="?", type=int, default=100,
                        help='size of the full data grid')

    # model params
    parser.add_argument('--model_name', nargs="?", type=str, default="lstm",
                        help='Name of the model used for the experiment or evaluation. \
                            Possible values: [lstm, keras_seq2seq, cnn_convlstm, windowed_cnn_convlstm, \
                            cnn_convlstm_seq2seq, cnn_convlstm_attention, predrnn, windowed_convlstm_seq2seq, \
                            convlstm_seq2seq]')
    parser.add_argument('--hidden_size', nargs="?", type=int, default=100, help='Hidden size')
    parser.add_argument('--num_layers', nargs="?", type=int, default=2, help='Number of layers')
    parser.add_argument('--encoder_filters', nargs="?", type=str2int_list, default="50", 
        help='Number of filters in each encoder layer. Format:  int,int,int ')
    parser.add_argument('--decoder_filters', nargs="?", type=str2int_list, default="50",
        help='Number of filters in each decoder layer. Format:  int,int,int ')
    parser.add_argument('--cnn_filters', nargs="?", type=str2int_list, default="25,50,50",
        help='Number of filters in each cnn layer. Format:  int,int,int ')
    parser.add_argument('--hidden_sizes', nargs="?", type=str2int_list, default="25,25",
        help='Number of filters in predrnn layers')
    parser.add_argument('--mlp_hidden_sizes', nargs="?", type=str2int_list, default="50,1",
        help='Hidden sizes in the mlp output layers')
    parser.add_argument('--decoder_padding', nargs="?", type=str, default="same",
        help='Padding type for the decoder (starting from layer 2)')
    parser.add_argument('--kernel_size', nargs="?", type=int, default=3, help='Convolutional kernel size')
    parser.add_argument('--pass_state', nargs="?", type=str2bool, default=True, 
        help='Determines if the decoder should receive the encoders state.')

    # experiment details params
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--data_path', nargs="?", type=str, default="data",
                        help='Path to the folder with the datasets')
    parser.add_argument('--gpus', nargs="?", type=int, default=0, 
        help="Number of gpus available")
    parser.add_argument('--model_file', nargs="?", type=str, default="none",
                        help='File path for the saved model')
    parser.add_argument('--create_tensorboard', nargs="?", type=str2bool, default=False, 
        help="A flag indicating whether logs for tensorboard should be collected (applicable only for certain models)")
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')

    # training params
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--use_mini_data', nargs="?", type=str2bool, default=False, 
        help="A flag indicating whether to use mini data set [use only for testing the setup]")
    parser.add_argument('--fraction_of_data', nargs="?", type=float, default=1,
                        help='Fraction of data to use for training')
    parser.add_argument('--fraction_of_val', nargs="?", type=float, default=1,
                        help='Fraction of validation samples to use for val evaluations')
    parser.add_argument('--learning_rate', nargs="?", type=float, default=1e-4,
                        help='Learning rate passed to the optimizer')
    parser.add_argument('--learning_rate_decay', nargs="?", type=float, default=0,
                        help='Learning rate decay (Adam)')
    parser.add_argument('--shuffle_order', nargs="?", type=str2bool, default=True, 
        help="A flag indicating whether to shuffle the data samples [use 'False' only for testing the setup]")
    parser.add_argument('--dropout', nargs="?", type=float, default=0, help='Dropout rate for the model')
    

    # Model evaluator specific arguments    
    parser.add_argument('--train_mean', nargs="?", type=float, default=67.61768898039853,
                        help='Mean of the initial training data')
    parser.add_argument('--train_std', nargs="?", type=float, default=132.47248595705986,
                        help='Standard deviation of the initial training data')
    parser.add_argument('--evaluation_steps', nargs="?", type=str2int_list, default="10,12",
                        help='list of number of steps to evaluate on')
    parser.add_argument('--prediction_batch_size', nargs="?", type=int, default="10000",
                        help='how many predictions to gather before computing the nrmse loss. \
                        10000 for single point predictions, 1 for full grid predictions')
    parser.add_argument('--missing_data', nargs="?", type=float, default=0,
                        help='Fraction of missing data for model evaluation')

    # unused
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    

    args = parser.parse_args()

    gpu_id = str(args.gpu_id)
    if gpu_id != "None":
        args.gpu_id = gpu_id



    print(args)
    return args
