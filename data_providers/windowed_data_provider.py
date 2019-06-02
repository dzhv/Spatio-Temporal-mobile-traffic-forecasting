import sys
from os import path
import os

parent_folder = path.dirname(path.abspath(__file__))
sys.path.append(parent_folder)

import numpy as np
import window_slider
DEFAULT_SEED = 20112018


class WindowedDataProvider(object):
    def __init__(self, data_reader, window_size=11, segment_size=12,
            batch_size=1000, shuffle_order=True, rng=None, fraction_of_data=1):
                
        self.window_size = window_size
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng        
        self.fraction_of_data = fraction_of_data

        self.data = data_reader.next()

        # number of time points which can be used to form inputs-targets pairs
        # the last segments are discarded as they cannot have a target
        self.num_segments = self.data.shape[0] - self.segment_size  
        
        # used only for progress bars
        self.num_batches = max(1, np.ceil(self.num_segments * self.fraction_of_data)) \
            * (self.data.shape[-1]**2 // batch_size)
            

    def next(self):
        indexes = self.rng.permutation(self.num_segments) if self.shuffle_order else np.arange(self.num_segments)
        return self.enumerate_data(indexes)

    def get_random_samples(self, n_samples):
        assert n_samples <= self.num_segments, f"Cannot provide more than {self.num_segments} samples"
        # returns samples from n_samples different starting time points
        indexes = self.rng.permutation(self.num_segments)[:n_samples]
        return self.enumerate_data(indexes)

    def enumerate_data(self, indexes):
        for count, i in enumerate(indexes):
            segment = self.data[i:i + self.segment_size + 1]  # +1 is to include the target grid

            inputs, targets = window_slider.get_windowed_segmented_data(
                segment, self.window_size, self.segment_size)

            assert inputs.shape[0] % self.batch_size == 0, f"batch_size needs to be a divider of {inputs.shape[0]}"

            if self.shuffle_order:
                perm = self.rng.permutation(inputs.shape[0])
                inputs = inputs[perm]
                targets = targets[perm]

            for batch_indx in range(0, inputs.shape[0], self.batch_size):
                yield (inputs[batch_indx:(batch_indx + self.batch_size)],
                    targets[batch_indx:(batch_indx + self.batch_size)])

            if count + 1 > len(indexes) * self.fraction_of_data:
                break


    def __iter__(self):
        for next_batch in self.next():
            yield next_batch

