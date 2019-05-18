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
            batch_size=10, segment_chunk_size=30, shuffle_order=True, rng=None):

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
        self.segment_chunk_size = segment_chunk_size
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng        
        self.data_reader = data_reader
        

    def next(self):
        read_index = 0

        remainder = None
        # for each file in the data_reader
        for file_data in self.data_reader.next():

            read_data = file_data if remainder is None else np.concatenate((remainder, file_data))

            file_size = file_data.shape[0]
            # process the file_data in chunks
            while read_index + self.segment_size <= file_size:

                # ensure the chunk is not bigger than the remaining data
                last_index = min(read_index + self.segment_chunk_size, file_size)
                segment_chunk = file_data[read_index:last_index, :, :]
                read_index = last_index - self.segment_size

                inputs, targets = window_slider.get_windowed_segmented_data(
                    segment_chunk, self.window_size, self.segment_size)

                
                # shuffle the content segment batch
                if self.shuffle_order:
                    perm = self.rng.permutation(inputs.shape[0])
                    inputs = inputs[perm]
                    targets = targets[perm]

                # here an assumption is made that inputs.shape[0] is a multiple of batch_size
                if inputs.shape[0] % self.batch_size != 0:
                    print("WARNING: number of datapoints in a chunk is not a multiple of batch_size, " +
                        "some datapoints will not be used for training/prediction.")

                batch_index = 0
                while batch_index + self.batch_size <= inputs.shape[0]:
                    yield inputs[batch_index:batch_index + self.batch_size], \
                        targets[batch_index:batch_index + self.batch_size]
                    batch_index += self.batch_size

            remainder = file_data[read_index:]


    def __iter__(self):
        for next_batch in self.next():
            yield next_batch

