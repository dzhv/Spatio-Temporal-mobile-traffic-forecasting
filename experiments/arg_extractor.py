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
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=1, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--segment_size', nargs="?", type=int, default=150000,
                        help='segment size of 1 training/validation sample')
    parser.add_argument('--element_size', nargs="?", type=int, default=1000,
                        help='size of the segment elements')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--data_path', nargs="?", type=str, default="data",
                        help='Path to the folder with the datasets')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--dropout', nargs="?", type=float, default=0,
                        help='Dropout rate for the model')
    parser.add_argument('--downsampled', nargs="?", type=str2bool, default=False,
                        help='Use downsampled dataset')
    parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    parser.add_argument('--learning_rate', nargs="?", type=float, default=1e-3,
                        help='Learning rate passed to the optimizer')

    parser.add_argument('--num_layers', nargs="?", type=int, default=2, help='Number of layers')
    parser.add_argument('--hidden_size', nargs="?", type=int, default=150, help='Hidden size')

    args = parser.parse_args()

    gpu_id = str(args.gpu_id)
    if gpu_id != "None":
        args.gpu_id = gpu_id



    print(args)
    return args
