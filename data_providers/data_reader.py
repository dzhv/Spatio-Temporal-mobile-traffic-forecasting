import sys
from os import path
import os
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(parent_folder)

import numpy as np

class SingleFileReader(object):
    def __init__(self, which_set='train'):
        assert which_set in ['train', 'valid', 'test'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        
        files = { 
            "train": "2013-11-01.npy",
            "valid": "2013-11-02.npy",
            "test": "2013-11-03.npy",
        }
        file = files.get(which_set)

        data_path = os.path.join(parent_folder, "data", file)
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        # load data from numpy file
        print(f"loading data from: {data_path}")
        self.loaded_data = np.load(data_path)
        print(f"loaded data with shape: {self.loaded_data.shape}")

    def next(self):
        return self.loaded_data