import numpy as np
from numpy.lib.stride_tricks import as_strided                 

# sliding_window_view taken from: https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a
def sliding_window_view(arr, window_shape, steps):
    """ Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.
        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.
        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]
        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]
        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1
        Notes
        -----
        In general, given
          `out` = sliding_window_view(arr,
                                      window_shape=[Wx, (...), Wz],
                                      steps=[Sx, (...), Sz])
           out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]
         Examples
         --------
         >>> import numpy as np
         >>> x = np.arange(9).reshape(3,3)
         >>> x
         array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
         >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
         >>> y
         array([[[[0, 1],
                  [3, 4]],
                 [[1, 2],
                  [4, 5]]],
                [[[3, 4],
                  [6, 7]],
                 [[4, 5],
                  [7, 8]]]])
        >>> np.shares_memory(x, y)
         True
        # Performing a neural net style 2D conv (correlation)
        # placing a 4x4 filter with stride-1
        >>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
        >>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
        >>> windowed_data = sliding_window_view(data,
        ...                                     window_shape=(4, 4),
        ...                                     steps=(1, 1))
        >>> conv_out = np.tensordot(filters,
        ...                         windowed_data,
        ...                         axes=[[1,2,3], [3,4,5]])
        # (F, H', W', N) -> (N, F, H', W')
        >>> conv_out = conv_out.transpose([3,0,1,2])
         """
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)

def get_windowed_data(data, window_size):
	"""
		INPUTS:
			data - a 3-dimensional numpy array (T, W, H)
				T - the time dimension
				W, H - width and height
			window_size - the size of the sliding window (int - assumed square)

		RETURNS:
			a 4-dimensional numpy array (K, T, W, H)
				K - dimension with all windows for one grid map
				T - the time dimension (remains the same as for input)
				W, H - window width and height
	"""
	assert window_size % 2 == 1, "window size should be odd"

	pad_amount = (window_size - 1) // 2
	padded_data = np.pad(data, [(0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount)], 'constant')

	# returns (1, number_of_windows, number_of_windows, T, window_size, window_size) size array
	windowed_data = sliding_window_view(padded_data, (padded_data.shape[0], window_size, window_size), 
		steps=[1, 1, 1])

	# remove the first dimension
	squeezed = np.squeeze(windowed_data)
	# print(squeezed.shape)

	# concat the (number_of_windows, number_of_windows) dimensions
	reshaped = squeezed.reshape(squeezed.shape[0]*squeezed.shape[1], squeezed.shape[2], 
	                       squeezed.shape[3], squeezed.shape[4])
	# print(reshaped.shape)

	return reshaped

	# make the T dimension as the first one
	# reordered = np.moveaxis(reshaped, 1, 0)
	# # print(reordered.shape)
	
	# return reordered


def get_windowed_segmented_data(data, window_size, segment_size):
	"""
		INPUTS:
			data - a 3-dimensional numpy array (T, W, H)
				T - the time dimension
				W, H - width and height
			window_size - size of the sliding window (int - assumed square)
			segment_size - length of temporal segments

		RETURNS:
			a touple (inputs, targets)
				where inputs is a 4-dimensional numpy array (N, S, W, H)
					N - number of data points (segments)
					S - segment length
					W, H - window width and height
				targets - a length N array with next time step amplitudes for each segment
	"""

	# (K, T, W, H)
	windowed_data = get_windowed_data(data, window_size)
	shape = windowed_data.shape

	# slide through the time axis and get segments
	segmented = sliding_window_view(windowed_data, 
		window_shape=[shape[0], segment_size, shape[2], shape[3]], steps=[1, 1, 1, 1])
	segmented = np.squeeze(segmented)

	reordered = np.moveaxis(segmented, 0, 1)
	# (10000, 133, 12, 11, 11)
	# (grid points, number of segments, segment length, window width, window height)
	# print(f"reorderd shape: {reordered.shape}")

	# dropping last segments as they will not have a target
	trimmed = reordered[:, :(reordered.shape[1] - 1), :, :, :]
	tshape = trimmed.shape
	# print(f"trimmed shape: {tshape}")

	reshaped = trimmed.reshape(tshape[0] * tshape[1], tshape[2], tshape[3], tshape[4])
	# print(f"reshaped shape: {reshaped.shape}")

	dshape = data.shape
	target_data = data[segment_size:, :, :]
	reordered_targets = np.transpose(target_data, (1, 2, 0))
	targets = reordered_targets.reshape((dshape[0] - segment_size) * dshape[1] * dshape[2])
	# print(f"targets shape: {targets.shape}")


	return reshaped, targets

def get_sequential_inputs_and_targets(data, window_size, segment_size, output_size):
  assert data.shape[0] == (segment_size + output_size), "Data should include input size + output size of data"

  windowed_data = get_windowed_data(data[:segment_size], window_size)
  shape = windowed_data.shape
  # print(f"windowed_data shape: {shape}")

  # slide through the time axis and get segments
  segmented = sliding_window_view(windowed_data, 
    window_shape=[shape[0], segment_size, shape[2], shape[3]], steps=[1, 1, 1, 1])
  inputs = np.squeeze(segmented)

  # (10000, 12, 11, 11)
  # print(f"inputs shape: {inputs.shape}")

  target_data = data[segment_size:]
  # print(f"target_data shape: {target_data.shape}")
  reshaped_targets = target_data.reshape(target_data.shape[0], target_data.shape[1] * target_data.shape[2])
  # print(f"reshaped_targets shape: {reshaped_targets.shape}")
  targets = reshaped_targets.T
  # print(f"targets shape: {targets.shape}")

  return inputs, targets
