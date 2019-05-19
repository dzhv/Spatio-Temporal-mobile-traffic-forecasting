import sys
from os import path
import os

import numpy as np

class DataReader(object):
    def __init__(self, data_folder, file_dict, which_set):
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or test. '
            'Got {0}'.format(which_set)
        )
        
        file = file_dict.get(which_set)

        data_path = os.path.join(data_folder, file)
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        # load data from numpy file
        print(f"loading data from: {data_path}")
        self.loaded_data = np.load(data_path)
        print(f"loaded data with shape: {self.loaded_data.shape}")

    def next(self):
        return self.loaded_data

class MiniDataReader(DataReader):
    def __init__(self, data_folder, which_set='train'):
        files = { 
            "train": "mini_train.npy",
            "valid": "mini_val.npy",
            "test": "mini_test.npy",
        }
        
        super(MiniDataReader, self).__init__(data_folder, files, which_set)    

class FullDataReader(DataReader):
    def __init__(self, data_folder, which_set='train'):
        files = { 
            "train": "train.npy",
            "valid": "val.npy",
            "test": "test.npy",
        }
        
        super(FullDataReader, self).__init__(data_folder, files, which_set)    
