from data_provider import DataProvider
import sys
from os import path
import os
import numpy as np

parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
# sys.path.append(parent_folder)

class MiniDataProvider(DataProvider):
    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )

        print(which_set)
        files = { 
            "train": "2013-12-01.npy",
            "valid": "2013-12-02.npy",
            "test": "2013-12-03.npy",
        }
        file = files.get(which_set)

        data_path = os.path.join(parent_folder, "data", "december_mapped", file)
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        print(f"loading data from: {data_path}")
        loaded = np.load(data_path)
        print(f"loaded data with shape: {loaded.shape}")

        
                
        super(MiniDataProvider, self).__init__(
             inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MiniDataProvider, self).next()
        return inputs_batch, targets_batch

dp = MiniDataProvider(which_set="valid")

