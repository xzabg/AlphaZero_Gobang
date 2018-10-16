# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

"""

import numpy as np
import tensorflow as tf


class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None, graph=None, output='./'):
        self.board_width = board_width
        self.board_height = board_height
        self.channel = 4
        self.output_dir = output
        self.n_step = 0
        self.training = tf.placeholder_with_default(True, shape=(), name='training')

        # Define the tensorflow neural network
        # 1. Input:
        with tf.name_scope('Input'):
            self.input_states = tf.placeholder(
                    tf.float32, shape=[None, self.channel, board_height, board_width])
            self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        # 2. Common Networks Layers
        with tf.name_scope('conv1'):
            self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                          filters=self.channel*8, kernel_size=[3, 3],
                                          padding="same", data_format="channels_last",
                                          activation=None)
            self.bn1 = tf.layers.batch_normalization(self.conv1, training=self.training, name='bn1')
            self.out1 = tf.nn.relu(self.bn1)

        with tf.name_scope('res1'):
            self.res1 = self.res_block(self.out1)

        with tf.name_scope('res2'):
            self.res2 = self.res_block(self.res1)

        with tf.name_scope('res3'):
            self.res3 = self.res_block(self.res2)

        # 3-1 Action Networks
        with tf.name_scope('action_conv'):
            self.action_conv = tf.layers.conv2d(inputs=self.res3, filters=4,
                                                kernel_size=[1, 1], padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        # Flatten the tensor
        with tf.name_scope('action_conv_flat'):
            self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        with tf.name_scope('action_fc'):
            self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                             units=board_height * board_width,
                                             activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        with tf.name_scope('evaluation_conv'):
            self.evaluation_conv = tf.layers.conv2d(inputs=self.res3, filters=2,
                                                    kernel_size=[1, 1],
                                                    padding="same",
                                                    data_format="channels_last",
                                                    activation=tf.nn.relu)
            self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        with tf.name_scope('evaluation_fc1'):
            self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                                  units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        with tf.name_scope('evaluation_fc2'):
            self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                                  units=1, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        with tf.name_scope('value_loss'):
            self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                           self.evaluation_fc2)
            tf.summary.scalar('loss', self.value_loss)
        # 3-2. Policy Loss function
        with tf.name_scope('policy_loss'):
            self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
            self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
            tf.summary.scalar('loss', self.policy_loss)
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        with tf.name_scope('l2_penalty'):
            l2_penalty = l2_penalty_beta * tf.add_n(
                [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        with tf.name_scope('loss'):
            self.loss = self.value_loss + self.policy_loss + l2_penalty
            tf.summary.scalar('loss', self.loss)

        # Define the optimizer we use for training
        with tf.name_scope('train'):
            self.learning_rate = tf.placeholder(tf.float32)
            tf.summary.scalar('learning rate', self.learning_rate)
            op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = op.minimize(self.loss)

        # Make a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=graph, config=config)
        self.merge = tf.summary.merge_all()
        self.filewriter = tf.summary.FileWriter(self.output_dir, self.session.graph)

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        self.saver = tf.train.Saver(var_list=var_list)
        if model_file is not None:
            self.restore_model(model_file)

        '''if model_file is not None:
            self.saver = tf.train.import_meta_graph(model_file + '.meta')
            self.restore_model(model_file)'''

    def res_block(self, input_layer):
        shortcut = input_layer
        x = input_layer
        with tf.name_scope('res_conv1'):
            x = tf.layers.conv2d(inputs=x,
                                 filters=self.channel*8, kernel_size=[3, 3],
                                 padding="same", data_format="channels_last",
                                 activation=None)
            x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.nn.relu(x)
        with tf.name_scope('res_conv2'):
            x = tf.layers.conv2d(inputs=x,
                                 filters=self.channel*8, kernel_size=[3, 3],
                                 padding="same", data_format="channels_last",
                                 activation=None)
            x = tf.layers.batch_normalization(x, training=self.training)

        return tf.nn.relu(x + shortcut)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch,
                           self.training: False}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, self.channel, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr,
                           self.training: True})
        if self.n_step % 30 == 0:
            summary = self.session.run(
                self.merge,
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr}
            )
            self.filewriter.add_summary(summary, self.n_step)
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)

    def __del__(self):
        print("session closed")
        self.session.close()
