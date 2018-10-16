# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.channel = int(4)
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((self.channel, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            for i in range(self.channel//2-1):
                square_state[self.channel-4-2*i][move_curr // self.width,
                                                 move_curr % self.height] = 1.0
                square_state[self.channel-3-2*i][move_oppo // self.width,
                                                 move_oppo % self.height] = 1.0
                if len(move_curr) > 0:
                    move_curr = np.delete(move_curr, -1)
                if len(move_oppo) > 0:
                    move_oppo = np.delete(move_oppo, -1)

            # indicate the last move location
            square_state[self.channel-2][self.last_move // self.width,
                                         self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[self.channel-1][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def revert_move(self, move):
        self.states.pop(move)
        self.availables.append(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

    '''def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row + 2:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1
    '''
    def count_on_direction(self, i, j, xdirection, ydirection, color):
        count = 0
        states = self.states
        for step in range(1, 5):
            if ydirection != 0 and (j + ydirection * step < 0 or j + ydirection * step >= self.width):
                break
            if xdirection != 0 and (i + xdirection * step < 0 or i + xdirection * step >= self.height):
                break
            if states.get((i + xdirection * step)*self.width + j + ydirection * step, -1) == color:
                count += 1
            else:
                break
        return count

    def has_a_winner(self):
        states = self.states
        n = self.n_in_row

        if self.last_move != -1:
            moved = self.last_move
        else:
            return False, -1

        player = states[moved]
        location = np.array(self.move_to_location(moved))

        directions = [[(-1, 0), (1, 0)],
                      [(0, -1), (0, 1)],
                      [(-1, 1), (1, -1)],
                      [(-1, -1), (1, 1)]]

        for axis in directions:
            axis_count = 1
            for (xdirection, ydirection) in axis:
                axis_count += self.count_on_direction(location[0], location[1], xdirection, ydirection, player)
                if axis_count >= n:
                    return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        # self.init_state()
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                        print(winner)
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        winners_z = np.array([])
        episode_len = 0
        n_games = 0
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)

            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            episode_len += 1
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners = np.zeros(episode_len)
                if winner != -1:
                    winners[::-2] = 1.0
                    winners[-2::-2] = -1.0
                winners_z = np.append(winners_z, winners)
                # reset MCTS root node
                episode_len = 0
                player.reset_player()
                self.board.init_board()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")

                if len(winners_z) > 3000:
                    return winner, zip(states, mcts_probs, winners_z)
                else:
                    n_games += 1
                    print("game {}".format(n_games))

    def init_state(self):
        '''locations = [[7, 7], [8, 8],
                     [6, 8], [8, 6],
                     [5, 7], [7, 9],
                     [9, 7], [8, 7],
                     [8, 5], [8, 9],
                     [8, 10], [10, 9],
                     [9, 9], [9, 8]]'''
        locations = [[0, 0], [7, 7],
                     [0, 14], [8, 8],
                     [14, 0], [9, 9]]
        for location in locations:
            move = self.board.location_to_move(location)
            self.board.do_move(move)
