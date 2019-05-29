import sys
from os import path
parent_folder = path.dirname(path.abspath(__file__))
sys.path.append(parent_folder)
import numpy as np
DEFAULT_SEED = 20112018

class FullGridDataProvider(object):
    def __init__(self, data_reader, segment_size=12, target_segment_size=1,
            batch_size=50, shuffle_order=True, rng=None, fraction_of_data=1):
                
        self.target_segment_size = target_segment_size
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.shuffle_order = shuffle_order
        self.rng = rng if not rng is None else np.random.RandomState(DEFAULT_SEED)
        self.fraction_of_data = fraction_of_data

        self.data = data_reader.next()

        # number of time points which can be used to form inputs-targets pairs
        # the last segments are discarded as they cannot have a target
        self.num_segments = self.data.shape[0] - self.segment_size - self.target_segment_size + 1
        
        # used only for progress bars
        self.num_batches = np.ceil(self.num_segments * self.fraction_of_data)

    def next(self):
        indexes = self.rng.permutation(self.num_segments) if self.shuffle_order else np.arange(self.num_segments)
        return self.enumerate_data(indexes)

    def enumerate_data(self, indexes):
        batch = None
        for count, i in enumerate(indexes):
            segment = self.data[i:i + self.segment_size + self.target_segment_size]

            start_of_targets = i + self.segment_size
            inputs = self.data[i: start_of_targets]
            targets = self.data[start_of_targets : start_of_targets + self.target_segment_size]

            # add to batch
            if batch is None:
                # adding the first (None) axis, so that later samples can be appended to the batch
                # along this axis
                batch = (inputs[None, :], targets[None, :])                
            else:
                batch = np.append(batch[0], inputs[None, :], axis=0), np.append(batch[1], targets[None, :], axis=0)

            if batch[0].shape[0] >= self.batch_size:
                yield batch
                batch = None

            if count + 1 > len(indexes) * self.fraction_of_data:
                break

    def get_random_samples(self, n_samples):
        assert n_samples <= self.num_segments, f"Cannot provide more than {self.num_segments} samples"
        # returns samples from n_samples different starting time points
        indexes = self.rng.permutation(self.num_segments)[:n_samples]
        return self.enumerate_data(indexes)


    def __iter__(self):
        for next_batch in self.next():
            yield next_batch

