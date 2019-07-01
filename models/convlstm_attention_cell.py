from keras.utils import conv_utils
import keras.backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine.base_layer import Layer
from keras.layers import Conv2D
import tensorflow as tf
import numpy as np
from keras.utils.generic_utils import object_list_uid


class ConvLSTMAttentionCell(Layer):
    """Cell class for the ConvLSTM2D layer.

    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al. (2015)](
            http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self, filters,
                 kernel_size,
                 attention_kernel_size=(3, 3),
                 attention_activation='tanh',
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ConvLSTMAttentionCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.attention_kernel_size = conv_utils.normalize_tuple(attention_kernel_size, 2, 'attention_kernel_size')
        self.attention_activation = activations.get(attention_activation)
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        # input that is specific for this cell at this time step
        # (None, window_size, window_size, num_channels)
        temporal_input_shape = input_shape[0]

        # full previous (encoder) layer output
        # (None, segment_size, window_size, window_size, num_channels)
        full_sequence_input_shape = input_shape[1]

        input_shape = temporal_input_shape

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)
        
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters,), *args, **kwargs),
                        initializers.Ones()((self.filters,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.filters * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :, :, :self.filters]
        self.recurrent_kernel_i = self.recurrent_kernel[:, :, :, :self.filters]
        self.kernel_f = self.kernel[:, :, :, self.filters: self.filters * 2]
        self.recurrent_kernel_f = (
            self.recurrent_kernel[:, :, :, self.filters: self.filters * 2])
        self.kernel_c = self.kernel[:, :, :, self.filters * 2: self.filters * 3]
        self.recurrent_kernel_c = (
            self.recurrent_kernel[:, :, :, self.filters * 2: self.filters * 3])
        self.kernel_o = self.kernel[:, :, :, self.filters * 3:]
        self.recurrent_kernel_o = self.recurrent_kernel[:, :, :, self.filters * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.filters]
            self.bias_f = self.bias[self.filters: self.filters * 2]
            self.bias_c = self.bias[self.filters * 2: self.filters * 3]
            self.bias_o = self.bias[self.filters * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

        # attention weights and biases

        # weight matrix applied to hidden state
        # self.filters defines the size of hidden state
        # input_dim defines number of filters for the kernel, which needs to be equal to encoder output
        self.W_ha = self.add_weight(shape=self.attention_kernel_size + (self.filters, input_dim),
            initializer=self.kernel_initializer,
            name='attention_hidden_kernel')

        # weight matrix applied to encoder outputs        
        self.W_xa = self.add_weight(shape=self.attention_kernel_size + (input_dim, input_dim),
            initializer=self.kernel_initializer,
            name='attention_input_kernel')

        self.bias_a = self.add_weight(shape=(input_dim,),
            name='attention_bias', 
            initializer=self.bias_initializer)  

        # weight matrix used to get unnormalized weights
        self.W_za = self.add_weight(shape=self.attention_kernel_size + (input_dim, input_dim),
            initializer=self.kernel_initializer,
            name='attention_weight_kernel')

        self.built = True

    def input_attention(self, encoder_outputs, h_state):
        # Z = W_z * tanh( W_xa * encoder_outputs + W_ha * H_state + b_a)
        # A = softmax(Z)
        # return sum_timewise( A (.) encoder_outputs )         

        print(f"h_state: {h_state}")
        
        hidden_convolution = self.input_conv(h_state, self.W_ha, b=self.bias_a, padding='same')
        print(f"hidden_convolution: {hidden_convolution}")

        outputs_convolution = self.timedistributted_convolution(encoder_outputs, self.W_xa)
        print(f"timedistributted out: {outputs_convolution}")        

        s = tf.expand_dims(hidden_convolution, axis=1) + outputs_convolution
        print(f"sum: {s}")

        act = self.attention_activation(s)
        print(f"activation: {act}")

        unnormalized_weights = self.timedistributted_convolution(act, self.W_za)
        print(f"unnormalized_weights: {unnormalized_weights}") 

        normalized_weights = tf.nn.softmax(unnormalized_weights, axis=1)
        print(f"normalized_weights: {unnormalized_weights}")        

        # elemntwise multiplication
        weighted_outputs = tf.multiply(normalized_weights, encoder_outputs)
        print(f"weighted_outputs: {weighted_outputs}")        

        weighted_sum = tf.reduce_sum(weighted_outputs, axis=1)
        print(f"weighted_sum: {weighted_sum}")

        return weighted_sum

    def call(self, inputs, states, training=None, constants=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        encoder_outputs = constants[0]

        inputs = self.input_attention(encoder_outputs, h_tm1)

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[1]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        x_i = self.input_conv(inputs_i, self.kernel_i, self.bias_i,
                              padding=self.padding)
        x_f = self.input_conv(inputs_f, self.kernel_f, self.bias_f,
                              padding=self.padding)
        x_c = self.input_conv(inputs_c, self.kernel_c, self.bias_c,
                              padding=self.padding)
        x_o = self.input_conv(inputs_o, self.kernel_o, self.bias_o,
                              padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i,
                                  self.recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f,
                                  self.recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c,
                                  self.recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o,
                                  self.recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h, c]

    def timedistributted_convolution(self, inputs, w, b=None):
        # applies convolution to a temporal sequence (batch_size, timesteps, height, width, channels)

        # print(f"inputs: {inputs}")

        # input_length = K.shape(inputs)[1]
        # inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)

        input_shape = K.int_shape(inputs)
        input_length = input_shape[1]
        # (-1, input_height, input_width, input_channels)
        inner_input_shape = (-1,) + input_shape[2:]
        # reshape inputs to (batch_size * timesteps, ...). 
        inputs = K.reshape(inputs, inner_input_shape)
        
        # print(f"inner_input_shape: {inner_input_shape}")
        # print(f"before conv: {inputs}")
        
        out = self.input_conv(inputs, w, b, padding='same')

        # print(f"after conv: {out}")
        
        # restore the original shape: (batch_size, timesteps, ...)
        # output_shape = self._get_shape_tuple((-1, input_length), out, 1, K.int_shape(out)[2:])
        output_shape = (-1, input_length) + (input_shape[2], input_shape[3], K.int_shape(w)[-1])

        # print(f"output shape: {output_shape}")
        out = K.reshape(out, output_shape)

        return out


    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
        """Finds non-specific dimensions in the static shapes
        and replaces them by the corresponding dynamic shapes of the tensor.

        # Arguments
            init_tuple: a tuple, the first part of the output shape
            tensor: the tensor from which to get the (static and dynamic) shapes
                as the last part of the output shape
            start_idx: int, which indicate the first dimension to take from
                the static shape of the tensor
            int_shape: an alternative static shape to take as the last part
                of the output shape

        # Returns
            The new int_shape with the first part from init_tuple
            and the last part from either `int_shape` (if provided)
            or K.int_shape(tensor), where every `None` is replaced by
            the corresponding dimension from K.shape(tensor)
        """
        # replace all None in int_shape by K.shape
        if int_shape is None:
            int_shape = K.int_shape(tensor)[start_idx:]
        if not any(not s for s in int_shape):
            return init_tuple + int_shape
        tensor_shape = K.shape(tensor)
        int_shape = list(int_shape)
        for i, s in enumerate(int_shape):
            if not s:
                int_shape[i] = tensor_shape[start_idx + i]
        return init_tuple + tuple(int_shape)