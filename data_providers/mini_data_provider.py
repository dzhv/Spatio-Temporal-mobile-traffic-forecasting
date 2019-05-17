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
            batch_size=10, segment_batch_size=30, shuffle_order=True, rng=None):
        

        """ Planas:
                nusiskaitai 1 dienos data
                is pirmu 30 padarai get_windowed_segmented_data
                    -> gauni 180000 segments
                juos susufflini ir atidavineji batchais
                kai baigiasi, imi nuo 19-o iki 49 ir darai get_windowed_segmented_data
                ... repeat
                kai baigiasi 144, skaitai naują failą.
        """
        self.window_size = window_size
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.segment_batch_size = segment_batch_size
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

        # used only for progress bars
        self.num_batches = (144 - self.segment_size) * 100 * 100 // batch_size

        self.loaded_data = data_reader.next()

    def next(self):
        read_index = 0

        while read_index + self.segment_batch_size <= self.loaded_data.shape[0]:
            segment_batch = self.loaded_data[read_index:read_index + self.segment_batch_size, :, :]
            read_index += self.segment_batch_size - self.segment_size

            inputs, targets = window_slider.get_windowed_segmented_data(
                segment_batch, self.window_size, self.segment_size)

            if self.shuffle_order:
                perm = self.rng.permutation(inputs.shape[0])
                inputs = inputs[perm]
                targets = targets[perm]

            batch_index = 0
            while batch_index + self.batch_size <= inputs.shape[0]:
                yield inputs[batch_index:batch_index + self.batch_size], \
                    targets[batch_index:batch_index + self.batch_size]
                batch_index += self.batch_size


    def __iter__(self):
        for next_batch in self.next():
            yield next_batch

