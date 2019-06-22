# Code adjusted from: https://github.com/Yunbo426/predrnn-pp

import sys
from os import path
parent_folder = path.dirname(path.dirname(path.abspath(__file__)))
grandparent_folder = path.dirname(parent_folder)
sys.path.append(parent_folder)
sys.path.append(grandparent_folder)
import numpy as np
import tensorflow as tf
from gradient_highway_unit import GHU as ghu
from causal_lstm_cell import CausalLSTMCell as cslstm
from models.model import Model

class PredRNN(Model):
    def __init__(self, batch_size=10, segment_size=12, output_size=1, window_size=11, hidden_size=50, 
        num_layers=2, learning_rate=0.001):

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

        self.mask_true = tf.placeholder(tf.float32,
                                        [batch_size,
                                         1,  # output_size - 1,  # output len - 1
                                         window_size,
                                         window_size,
                                         1])

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        # num_hidden = [int(x) for x in args.hidden_size.split(',')]
        num_hidden = [hidden_size for _ in range(num_layers)]
        # print(num_hidden)
        # num_layers = len(num_hidden)
        num_layers = num_layers
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = rnn(
                self.x,
                self.mask_true,
                num_layers, num_hidden,
                5, 1,   # filter size, stride
                segment_size + output_size, segment_size)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:,segment_size-1:]
            self.loss_train = loss / batch_size
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)
        # if FLAGS.pretrained_model:
        #     self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, x, y):
        # if data_provider is setup correctly:
        # x.shape == (batch_size, segment_size, window_size, window_size)
        # y.shape == (batch_size, 1, window_size, window_size)

        # concatenate x, y on timewise axis as this is the expected input for predrnn model
        inputs = np.concatenate((x, y), axis=1)
        # add empty channel dimension
        inputs = np.expand_dims(inputs, axis=-1)

        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: self.learning_rate})

        mask_true = np.zeros((self.batch_size,
            1, #self.output_size -1,
            self.window_size,
            self.window_size,
            1)) # channel
        feed_dict.update({self.mask_true: mask_true})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, path):
        checkpoint_path = path + '.ckpt'
        self.saver.save(self.sess, checkpoint_path)
        print('saved to ' + path)


def rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1, 
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
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            if t < input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-input_length]*images[:,t] + (1-mask_true[:,t-input_length])*x_gen

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
            gen_images.append(x_gen)

    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    loss = tf.nn.l2_loss(gen_images - images[:,1:])
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]

if __name__ == '__main__':
    batch_size = 2
    print("Lets build it")
    model = PredRNN(batch_size=batch_size)
    x = np.random.randn(batch_size, 12, 11, 11)
    y = np.random.randn(batch_size, 1, 11, 11)

    print("Lets train it")
    model.train(x, y)

    print("Sucess")