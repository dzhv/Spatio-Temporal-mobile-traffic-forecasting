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

class PredRNN(Model):
    def __init__(self, batch_size=1, segment_size=12, output_size=1, window_size=100, hidden_sizes=[50, 50], 
        learning_rate=0.001, dropout=0):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.segment_size = segment_size
        self.output_size = output_size
        self.window_size = window_size

        # inputs
        self.x = tf.placeholder(tf.float32,
                                [batch_size,
                                 segment_size + output_size,  # input + output length (?)
                                 window_size,
                                 window_size,
                                 1])

        self.training = tf.placeholder(tf.bool, shape=())

        self.mask_shape = [batch_size, max(1, output_size - 1), window_size, window_size, 1]
        self.mask_true = tf.placeholder(tf.float32, self.mask_shape)

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        
        num_layers = len(hidden_sizes)
        num_layers = num_layers
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = rnn(
                self.x,
                self.mask_true,
                num_layers, hidden_sizes,
                5, 1,   # filter size, stride
                segment_size + output_size, segment_size,
                dropout=dropout, training=self.training)
            pred_ims = output_list[0]            

            loss = output_list[1]

            self.loss_train = loss / batch_size
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables, max_to_keep=1000)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)

    def train(self, x, y):
        inputs, mask_true = self.prepare_inputs(x, y)

        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: self.learning_rate})
        feed_dict.update({self.training: True})

        feed_dict.update({self.mask_true: mask_true})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def forward(self, x):
        inputs, mask_true = self.prepare_inputs(x)

        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        feed_dict.update({self.training: False})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)

        assert len(gen_ims) == 1, f"this was supposed to be of length 1, was: {len(gen_ims)}"
        return gen_ims[0]

    def evaluate(self, x, y):
        inputs, mask_true = self.prepare_inputs(x, y)

        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        feed_dict.update({self.training: False})
        loss = self.sess.run(self.loss_train, feed_dict)

        return loss

    def prepare_inputs(self, x, y=None):
        # if data_provider is setup correctly:
        # x.shape == (batch_size, segment_size, window_size, window_size)
        # y.shape == (batch_size, 1, window_size, window_size)

        if y is None:
            y = np.zeros((self.batch_size, self.output_size, self.window_size, self.window_size))

        # concatenate x, y on timewise axis as this is the expected input for predrnn model
        inputs = np.concatenate((x, y), axis=1)
        # add empty channel dimension
        inputs = np.expand_dims(inputs, axis=-1)
                
        # setting mask_true to zeros corresponds to always using model predictions
        # as inputs for future predictions
        mask_true = np.zeros(self.mask_shape)

        return inputs, mask_true

    def save(self, path):
        checkpoint_path = path + '.ckpt'
        self.saver.save(self.sess, checkpoint_path)
        print('saved to ' + path)

    def load(self, path):
        weight_path = path + '.ckpt'
        print(f"Loading weights from {weight_path}\n")
        self.saver.restore(self.sess, weight_path)


def rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1, 
    seq_length=20, input_length=10, tln=True, dropout=0, training=False):
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
                inputs = mask_true[:,t-input_length]*images[:,t] + (1-mask_true[:,t-input_length])*x_gen

            inputs = tf.keras.layers.Dropout(dropout)(inputs, training=training)
            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            z_t = tf.keras.layers.Dropout(dropout)(z_t, training=training)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, num_layers):
                inputs = tf.keras.layers.Dropout(dropout)(hidden[i-1], training=training)
                hidden[i], cell[i], mem = lstm[i](inputs, hidden[i], cell[i], mem)

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
    loss = tf.reduce_mean(tf.squared_difference(gen_images, images[:,input_length:]))
    # loss = tf.losses.mean_squared_error(labels=, predictions=gen_images)
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]

if __name__ == '__main__':
    batch_size = 2
    window_size = 6
    segment_size = 12    
    print("Lets build it")
    model = PredRNN(batch_size=batch_size, output_size=12, window_size=window_size)
    x = np.random.randn(batch_size, segment_size, window_size, window_size)
    y = np.random.randn(batch_size, segment_size, window_size, window_size)

    def try_outputs():
        print("\nLets train it\n")
        model.train(x, y)

        print("\nLets predict\n")
        out = model.forward(x)
        print(f"out shape: {np.array(out).shape}")

        print(f"model.pred_ims: {model.pred_seq}")
        out = model.forward(x + 2)
        out = model.forward(x + 5)
        print(f"model.pred_ims: {model.pred_seq}")

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