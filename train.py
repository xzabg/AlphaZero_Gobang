# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

"""

from __future__ import print_function
import os
import random
import time
import threading
import numpy as np
import tensorflow as tf
from collections import defaultdict, deque
from game import Board, Game
from sgf_wrapper import Manual
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras


class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.manual = Manual(self.board)
        # training params
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 100  # num of simulations for each move
        self.c_puct = 1
        self.buffer_size = 100000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.episode_len = 0
        self.kl_targ = 0.02
        self.check_freq = 1
        self.game_batch_num = 5
        self.best_win_ratio = 0.55
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.lock = threading.Lock()
        if init_model:
            # start training from an initial policy-value net
            self.g1 = tf.Graph()
            with self.g1.as_default():
                self.policy_value_net = PolicyValueNet(self.board_width,
                                                       self.board_height,
                                                       model_file=init_model,
                                                       graph=self.g1,
                                                       output='/data/data/')
            # tf.reset_default_graph()
            self.g2 = tf.Graph()
            with self.g2.as_default():
                self.policy_value_net_train = PolicyValueNet(self.board_width,
                                                             self.board_height,
                                                             model_file=init_model,
                                                             graph=self.g2,
                                                             output='/data/output/')
        else:
            # start training from a new policy-value net
            self.g1 = tf.Graph()
            with self.g1.as_default():
                self.policy_value_net = PolicyValueNet(self.board_width,
                                                       self.board_height,
                                                       graph=self.g1,
                                                       output='./data/')
            # tf.reset_default_graph()
            self.g2 = tf.Graph()
            with self.g2.as_default():
                self.policy_value_net_train = PolicyValueNet(self.board_width,
                                                             self.board_height,
                                                             graph=self.g2,
                                                             output='./output/')

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            # self.lock.acquire()
            # print("game {}".format(i))
            with self.g1.as_default():
                '''mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout,
                                         is_selfplay=1)
                board = Board(width=self.board_width,
                              height=self.board_height,
                              n_in_row=self.n_in_row)
                game = Game(board)'''
                winner, play_data = self.game.start_self_play(self.mcts_player,
                                                              is_shown=0,
                                                              temp=self.temp)
            # self.lock.release()

            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

        # print("self play end...")

    def collect_manual_data(self, file):
        winner, play_data = self.manual.read_manual_data(file)
        # read the chess manual fail
        if winner == 0:
            return

        play_data = list(play_data)[:]
        self.episode_len = len(play_data)
        # augment the data
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)

    def collect_test_data(self):
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        move = 128
        self.board.do_move(112)
        states.append(self.board.current_state())
        probs = np.zeros(self.board.width * self.board.height)
        probs[[move]] = 1
        mcts_probs.append(probs)
        current_players.append(self.board.current_player)
        winners_z = np.array([1])
        play_data = zip(states, mcts_probs, winners_z)
        play_data = list(play_data)[:]
        self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        with self.g2.as_default():
            for i in range(self.epochs):
                loss, entropy = self.policy_value_net_train.train_step(
                        state_batch,
                        mcts_probs_batch,
                        winner_batch,
                        self.learn_rate*self.lr_multiplier)

        print((
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               ).format(
                        self.lr_multiplier,
                        loss,
                        entropy))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        print("evaluating...")
        current_mcts_player = MCTSPlayer(self.policy_value_net_train.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.pure_mcts_playout_num)
        best_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          best_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))

        # save the current_model
        self.policy_value_net_train.save_model('/data/output/current_policy.model')
        if win_ratio > self.best_win_ratio:
            print("New best policy!!!!!!!!")
            # update the best_policy
            self.policy_value_net_train.save_model('/data/output/best_policy.model')
            self.g1 = tf.Graph()
            with self.g1.as_default():
                self.policy_value_net = PolicyValueNet(self.board_width,
                                                       self.board_height,
                                                       model_file='/data/output/best_policy.model',
                                                       graph=self.g1,
                                                       output='/data/data/')

        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            '''coord = tf.train.Coordinator()
            self_play = [threading.Thread(target=self.collect_selfplay_data, args=(self.play_batch_size,)) for i in range(4)]
            for sp in self_play:
                sp.start()
            coord.join(self_play)
            while len(self.data_buffer) < self.batch_size:
                print(len(self.data_buffer))
                time.sleep(3)
                pass'''
            multiplier = [0.1, 0.1, 0.01, 0.01, 0.01]
            step = 0
            for n in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                # self.collect_test_data()
                self.policy_value_net.n_step += 1

                print("batch i:{}, episode_len:{}".format(
                   self.policy_value_net.n_step, self.episode_len))

                # optimisation
                if len(self.data_buffer) > self.batch_size:
                    for i in range(100):
                        self.policy_update()

                # evaluation
                if self.policy_value_net.n_step % self.check_freq == 0:
                    # self.lr_multiplier = multiplier[step]
                    # step += 1
                    self.mcts_player.mcts._discount = 1 - 0.98*(1 - self.mcts_player.mcts._discount)
                    print("current self-play batch: {}, discount: {}".format(
                        self.policy_value_net.n_step, self.mcts_player.mcts._discount))

                    # self.lock.acquire()
                    self.policy_evaluate(n_games=15)
                    # self.lock.release()
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline('/data/data/current_policy.model')  # /data/data'models/sl-9/current_policy.model'
    training_pipeline.run()
