import sys
from os import path
import os

parent_folder = path.dirname(path.abspath(__file__))
sys.path.append(parent_folder)

import numpy as np
import window_slider
DEFAULT_SEED = 20112018


class MiniDataProvider(object):
    def __init__(self, data_reader, window_size=11, segment_size=12,
            batch_size=10, shuffle_order=True, rng=None):
                
        self.window_size = window_size
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        num_timesteps = 61   # limiting the data for this *mini* data provider
        
        # used only for progress bars
        self.num_batches = (num_timesteps - self.segment_size) * 100 * 100 // batch_size

        self.data = data_reader.next()[:num_timesteps]

    def next(self):
        # discarding last segments, which will not have a target
        num_segments = self.data.shape[0] - self.segment_size  

        indexes = self.rng.permutation(num_segments) if self.shuffle_order else np.arange(num_segments)

        for i in indexes:
            segment = self.data[i:i + self.segment_size + 1]  # +1 is to include the target grid

            inputs, targets = window_slider.get_windowed_segmented_data(
                segment, self.window_size, self.segment_size)

            print(f"inputs shape: {inputs.shape}")
            print(f"targets shape: {targets.shape}")

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

    # def next(self):
    #     read_index = 0

    #     while read_index + self.segment_chunk_size <= self.loaded_data.shape[0]:
    #         segment_chunk = self.loaded_data[read_index:read_index + self.segment_chunk_size, :, :]
    #         read_index += self.segment_chunk_size - self.segment_size

    #         inputs, targets = window_slider.get_windowed_segmented_data(
    #             segment_chunk, self.window_size, self.segment_size)

    #         if self.shuffle_order:
    #             perm = self.rng.permutation(inputs.shape[0])
    #             inputs = inputs[perm]
    #             targets = targets[perm]

    #         batch_index = 0
    #         while batch_index + self.batch_size <= inputs.shape[0]:
    #             yield inputs[batch_index:batch_index + self.batch_size], \
    #                 targets[batch_index:batch_index + self.batch_size]
    #             batch_index += self.batch_size

