import tensorflow as tf

from txtgen.losses.dqn_losses import l2_loss
from txtgen.core import optimization as opt


class MLPNetwork:
    def __init__(self, input_dimension, output_dimension, hidden_list=None,
                 loss_fn=None, train_hparams=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.layers = []
            self.layers.append(tf.placeholder(dtype=tf.float64, name='state_input', shape=(None, input_dimension)))

            if hidden_list is None:
                hidden_list = [128, 128]
            for i in range(len(hidden_list)):
                self.layers.append(tf.contrib.layers.fully_connected(inputs=self.layers[-1], num_outputs=hidden_list[i]))
            self.layers.append(
                tf.contrib.layers.fully_connected(inputs=self.layers[-1], num_outputs=output_dimension,
                                                  activation_fn=None))
            if loss_fn is None:
                loss_fn = l2_loss
            self.y_input = tf.placeholder(dtype=tf.float64, name='y_input', shape=(None, ))
            self.action_input = tf.placeholder(dtype=tf.float64, name='action_input', shape=(None, output_dimension))
            loss = loss_fn(q_value=self.layers[-1], y_input=self.y_input, action_input=self.action_input)

            if train_hparams is None:
                train_hparams = opt.default_optimization_hparams()
            self.trainer = opt.get_train_op(loss=loss, hparams=train_hparams)

        self.sess = tf.Session(graph=self.graph)
        self.initialize()

    def graph(self):
        return self.graph

    def session(self):
        return self.sess

    def initialize(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def get_params(self):
        param_dict = {}
        with self.graph.as_default():
            for u in tf.global_variables():
                param_dict[u.name] = self.sess.run(u)
        return param_dict

    def set_params(self, param_dict):
        with self.graph.as_default():
            for u in tf.global_variables():
                if u.name in param_dict:
                    self.sess.run(tf.assign(u, param_dict[u.name]))

    def get_qvalue(self, state_batch=None):
        return self.sess.run(self.layers[-1], feed_dict={'state_input:0': state_batch})

    def train(self, feed_dict=None):
        self.sess.run(self.trainer, feed_dict=feed_dict)
