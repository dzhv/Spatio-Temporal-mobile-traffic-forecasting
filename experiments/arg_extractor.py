import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', nargs="?", type=int, default=1000, help='Batch_size for experiment')
    parser.add_argument('--segment_size', nargs="?", type=int, default=12,
                        help='segment size of 1 training/validation sample')
    parser.add_argument('--window_size', nargs="?", type=int, default=11,
                        help='size of the segment elements')
    parser.add_argument('--hidden_size', nargs="?", type=int, default=100, help='Hidden size')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--data_path', nargs="?", type=str, default="data",
                        help='Path to the folder with the datasets')
    parser.add_argument('--use_mini_data', nargs="?", type=str2bool, default=False, 
        help="A flag indicating whether to use mini data set [use only for testing the setup]")
    parser.add_argument('--gpus', nargs="?", type=int, default=0, 
        help="Number of gpus available")
    parser.add_argument('--num_layers', nargs="?", type=int, default=2, help='Number of layers')
    parser.add_argument('--train_mean', nargs="?", type=float, default=67.61768898039853,
                        help='Mean of the initial training data')
    parser.add_argument('--train_std', nargs="?", type=float, default=132.47248595705986,
                        help='Standard deviation of the initial training data')
    parser.add_argument('--model_file', nargs="?", type=str, default="none",
                        help='File path for the saved model')
    parser.add_argument('--fraction_of_data', nargs="?", type=float, default=1,
                        help='Fraction of data to use for training')
    parser.add_argument('--learning_rate', nargs="?", type=float, default=1e-4,
                        help='Learning rate passed to the optimizer')
    parser.add_argument('--shuffle_order', nargs="?", type=str2bool, default=True, 
        help="A flag indicating whether to shuffle the data samples [use 'False' only for testing the setup]")

    # Model evaluator specific arguments
    parser.add_argument('--model_name', nargs="?", type=str, default="lstm",
                        help='Name of the model being evaluated. Possible values: [lstm, seq2seq]')

    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--dropout', nargs="?", type=float, default=0,
                        help='Dropout rate for the model')
    parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    

    args = parser.parse_args()

    gpu_id = str(args.gpu_id)
    if gpu_id != "None":
        args.gpu_id = gpu_id



    print(args)
    return args
