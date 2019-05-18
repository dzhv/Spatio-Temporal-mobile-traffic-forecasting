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
            batch_size=1000, shuffle_order=True, rng=None):
                
        self.window_size = window_size
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng        
        
        self.data = data_reader.next()

        # used only for progress bars
        self.num_batches = (self.data.shape[0] - self.segment_size) * self.data.shape[-1]**2 // batch_size

    def next(self):
        # discarding last segments, which will not have a target
        num_segments = self.data.shape[0] - self.segment_size  

        indexes = self.rng.permutation(num_segments) if self.shuffle_order else np.arange(num_segments)

        for i in indexes:
            segment = self.data[i:i + self.segment_size + 1]  # +1 is to include the target grid

            inputs, targets = window_slider.get_windowed_segmented_data(
                segment, self.window_size, self.segment_size)

            # print(f"inputs shape: {inputs.shape}")
            # print(f"targets shape: {targets.shape}")

            assert inputs.shape[0] % self.batch_size == 0, f"batch_size needs to be a divider of {inputs.shape[0]}"

            if self.shuffle_order:
                perm = self.rng.permutation(inputs.shape[0])
                inputs = inputs[perm]
                targets = targets[perm]

            for batch_indx in range(0, inputs.shape[0], self.batch_size):
                yield (inputs[batch_indx:(batch_indx + self.batch_size)],
                    targets[batch_indx:(batch_indx + self.batch_size)])


    def __iter__(self):
        for next_batch in self.next():
            yield next_batch

