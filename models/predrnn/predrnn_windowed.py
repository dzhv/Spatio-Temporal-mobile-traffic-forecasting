# Code adjusted from: https://github.com/Yunbo426/predrnn-pp

import sys
from os import path
this_folder = path.dirname(path.abspath(__file__))
parent_folder = path.dirname(this_folder)
grandparent_folder = path.dirname(parent_folder)
sys.path.append(this_folder)
sys.path.append(parent_folder)
sys.path.append(grandparent_folder)
import numpy as np
import tensorflow as tf
from gradient_highway_unit import GHU as ghu
from causal_lstm_cell import CausalLSTMCell as cslstm
from models.model import Model
import keras.backend as K

class PredRnnWindowed(Model):
    def __init__(self, batch_size=1, segment_size=12, output_size=12, window_size=11, 
        hidden_sizes=[25, 25], mlp_hidden_sizes=[50, 1], learning_rate=0.001, learning_rate_decay=0):

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.train_iterations = 0
        self.batch_size = batch_size
        self.segment_size = segment_size
        self.output_size = output_size
        self.window_size = window_size

        # inputs
        self.x = tf.placeholder(tf.float32,
                                [batch_size,
                                 segment_size,
                                 window_size,
                                 window_size,
                                 1], name='x_placeholder')

        self.y = tf.placeholder(tf.float32,
                                [batch_size,
                                 output_size,
                                 1], name='y_placeholder')

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        num_layers = len(hidden_sizes)
        num_layers = num_layers
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            out = rnn(
                self.x,
                num_layers, hidden_sizes,
                5, 1,   # filter size, stride
                segment_size + output_size, segment_size)
            print("!")
            print(out)
            print("!")
            predictions = self.timedistributted_mlp(out, mlp_hidden_sizes)

            loss = tf.reduce_mean(tf.squared_difference(predictions, self.y))

            self.loss_train = loss / batch_size
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))

            self.pred_seq.append(predictions)

        self.train_op = tf.train.AdamOptimizer(self.tf_lr).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables, max_to_keep=1000)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)

    def timedistributted_mlp(self, inputs, hidden_sizes):
        # applies mlp to a temporal sequence (batch_size, timesteps, height, width, channels)

        # print(f"inputs: {inputs}")
        input_shape = K.int_shape(inputs)
        input_length = input_shape[1]

        
        inner_input_shape = (-1, input_shape[2] * input_shape[3] * input_shape[4])
        # reshape inputs to (batch_size * timesteps, ...). 
        inputs = K.reshape(inputs, inner_input_shape)
        
        # print(f"inner_input_shape: {inner_input_shape}")
        # print(f"before conv: {inputs}")
        
        out = self.mlp(inputs, hidden_sizes)

        # print(f"after conv: {out}")
        
        # restore the original shape: (batch_size, timesteps, ...)
        # output_shape = self._get_shape_tuple((-1, input_length), out, 1, K.int_shape(out)[2:])
        output_shape = (-1, input_length, hidden_sizes[-1])

        # print(f"output shape: {output_shape}")
        out = K.reshape(out, output_shape)

        return out

    def mlp(self, inputs, hidden_sizes):
        out = inputs
        for i, h in enumerate(hidden_sizes):            
            layer = tf.layers.Dense(h, 
                activation= None if i == len(hidden_sizes) - 1 else tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer())

            out = layer(out)

        return out

    def train(self, x, y):
        lr = self.learning_rate * (1. / (1. + self.learning_rate_decay * self.train_iterations))
        self.train_iterations += 1

        feed_dict = {
            self.x: self.prepare_inputs(x),
            self.y: self.prepare_targets(y),
            self.tf_lr: self.learning_rate
        }
        
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)

        return loss

    def forward(self, x):
        feed_dict = {self.x: self.prepare_inputs(x)}
        gen_ims = self.sess.run(self.pred_seq, feed_dict)

        assert len(gen_ims) == 1, f"this was supposed to be of length 1, was: {len(gen_ims)}"
        return gen_ims[0]

    def evaluate(self, x, y):        
        feed_dict = {
            self.x: self.prepare_inputs(x),
            self.y: self.prepare_targets(y),
        }
        loss = self.sess.run(self.loss_train, feed_dict)

        return loss

    def prepare_inputs(self, x):
        # x.shape == (batch_size, segment_size, window_size, window_size)
        # adding an empty (channel) demension to the end
        return np.expand_dims(x, axis=-1)

    def prepare_targets(self, y):
        # y.shape  == (batch_size, segment_size)
        # adding an empty (channel) demension to the end
        return np.expand_dims(y, axis=-1)        

    def save(self, path):
        checkpoint_path = path + '.ckpt'
        self.saver.save(self.sess, checkpoint_path)
        print('saved to ' + path)

    def load(self, path):
        weight_path = path + '.ckpt'
        print(f"Loading weights from {weight_path}\n")
        self.saver.restore(self.sess, weight_path)


def rnn(images, num_layers, num_hidden, filter_size, stride=1, 
    seq_length=20, input_length=10, tln=True):
    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]

    for i in range(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = cslstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None
    
    for t in range(seq_length-1):
        with tf.variable_scope('predrnn_pp', reuse= not t == 0):
            if t < input_length:
                inputs = images[:,t]
            else:
                inputs = x_gen

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)

            x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name="back_to_pixel")

            if t >= input_length - 1:   # if we are already predicting
                gen_images.append(x_gen)

    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
        
    return gen_images

if __name__ == '__main__':
    batch_size = 2
    segment_size = 5
    output_size = 6
    print("Lets build it")
    model = PredRnnWindowed(batch_size=batch_size, segment_size=segment_size, output_size=output_size)
    x = np.random.randn(batch_size, segment_size, 11, 11)
    y = np.random.randn(batch_size, output_size)

    def try_outputs():
        print("\nLets train it\n")
        model.train(x, y)

        print("\nLets predict\n")
        out = model.forward(x)
        print(f"out shape: {np.array(out).shape}")

    path = 'temp/test'
    def try_saving():
        model.train(x, y)
        model.save(path)

    def try_loading():
        model.load(path)
        model.forward(x+2)
        model.train(x+1, y+3)


    try_outputs()

    print("Success")