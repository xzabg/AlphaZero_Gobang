# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

"""

import numpy as np
import copy
import time
from collections import deque
import threading
import tensorflow as tf


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def visit_to_prob(x, t=0):
    if t == 0:
        return softmax(x)
    else:
        vsum = np.sum(x)
        probs = x / vsum
        return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, discount=0.95):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self._discount = discount
        self.lr = 0.02
        self._sigma = 0.2
        self._restrict_range = 2
        self.width = 15
        self.height = 15
        self._select = 0
        if parent is None:
            self._move_restrict = [self.width, 0, self.height, 0]  # left, right, bottom, top
        else:
            self._move_restrict = copy.deepcopy(parent.get_range())

    def expand(self, action_priors, discount=0.95):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:  # and self.with_range(action):
                # TODO: expand restrict range
                self._children[action] = TreeNode(self, prob, discount=discount)
                # self._children[action].expand_range(action)
                self.action = action

    def compact(self, action):
        actions = list(self._children.keys())
        for act in actions:
            if act != action:
                self._children.pop(act)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
        '''max_value = -1
        selected_node = 0
        for node in list(self._children):
            value = self._children[node].get_value(c_puct)
            if value > max_value:
                selected_node = node
                max_value = value
        return selected_node, self._children[selected_node]'''

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # self._Q += self.lr * (leaf_value - self._Q)

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            discount = 1
            # discount = math.exp(-(1 - self._P) * (1 - self._P) / self._sigma)  # update 1
            # discount = 1 - math.exp(-self._P * self._P / self._sigma)  # update 2
            self._parent.update_recursive(-leaf_value * discount)
            # try to update the parent only to see if it would converge faster
            # self._parent.update(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def get_range(self):
        return self._move_restrict

    def with_range(self, action):
        if self._move_restrict == [self.width, 0, self.height, 0]:
            return True

        index_x = action // self.width
        index_y = action % self.width

        if self._move_restrict[0] <= index_x <= self._move_restrict[1] \
                and self._move_restrict[2] <= index_y <= self._move_restrict[3]:
            return True
        else:
            # print('out of range')
            return False

    def expand_range(self, action):
        self._move_restrict[0] = max(min(self._move_restrict[0], action // self.width - self._restrict_range), 0)
        self._move_restrict[1] = min(max(self._move_restrict[1], action // self.width + self._restrict_range), self.width - 1)
        self._move_restrict[2] = max(min(self._move_restrict[2], action % self.width - self._restrict_range), 0)
        self._move_restrict[3] = min(max(self._move_restrict[3], action % self.width + self._restrict_range), self.height - 1)
        # print('move range', self._move_restrict)


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._ROOT = self._root
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._discount = 0.95
        self._n_moves = 0
        self._n_beginning = 5
        self.start_time = 0
        self.total_time = 10
        self.n_threads = 64
        self.playout_queue = deque(maxlen=225)
        self.lock = threading.Lock()

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs, self._discount)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
                # leaf_value = 1
                # delete all the brothers of this node
                # node._parent.compact(action)

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def multi_playout(self, state, n_thread):
        while self.time_left() > 0:
            state_copy = copy.deepcopy(state)
            node = self._root
            while True:
                if node.is_leaf():
                    break
                action, node = node.select(self._c_puct)
                state_copy.do_move(action)

            action_probs, leaf_value = self._policy(state_copy)
            # Check for end of game.
            end, winner = state_copy.game_end()
            if not end:
                self.lock.acquire()
                node.expand(action_probs, self._discount)
                self.lock.release()
            else:
                # for end state，return the "true" leaf_value
                if winner == -1:  # tie
                    leaf_value = 0.0
                else:
                    leaf_value = (
                        1.0 if winner == state_copy.get_current_player() else -1.0
                    )
            node._Q = -leaf_value
            self.playout_queue.append(node)
            if len(self.playout_queue) == n_thread:
                node = self.playout_queue.popleft()
                self.lock.acquire()
                node.update_recursive(node._Q)
                self.lock.release()

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        '''self.start_time = time.time()
        self.playout_queue.clear()
        threads = []
        coord = tf.train.Coordinator()
        for i in range(self.n_threads):
            state_copy = copy.deepcopy(state)
            threads += [threading.Thread(target=self.multi_playout, args=(state_copy, self.n_threads))]
            threads[i].start()
        coord.join(threads)'''

        if self._root._select == 0:
            for n in range(self._n_playout):
                state_copy = copy.deepcopy(state)
                self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            # denote selected
            self._root._select = 1
            self._root = self._root._children[last_move]
            # self._root._parent = None
        else:
            # self._root = TreeNode(None, 1.0)
            self._root = self._ROOT

    def time_left(self):
        return self.total_time - time.time() + self.start_time

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            if not self._is_selfplay:
                self.mcts.update_with_move(board.last_move)
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                if len(board.states) > 3:
                    probs = softmax(1000*probs)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.03*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(move)
                # location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))
                # print("state value: %.3f" % self.mcts._root._Q)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
