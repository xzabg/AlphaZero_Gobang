# coding: utf-8
from game import Board, Game
import numpy as np
import sgf
import os


class Manual(object):

    def __init__(self, board):
        self.board = board
        self.game = Game(board)

    def read_manual_data(self, filename):
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        try:
            with open(filename) as f:
                collection = sgf.parse(f.read())
                for child in collection.children:
                    for node in child.nodes:
                        if len(node.properties) > 1:
                            if node.properties['RE'] == ['黑胜']:
                                winner = 1
                            elif node.properties['RE'] == ['白胜']:
                                winner = 2
                            else:
                                winner = -1
                            print('winner: ', winner)
                        elif len(node.properties) == 1:
                            if 'B' in node.properties:
                                loc = node.properties['B']
                            elif 'W' in node.properties:
                                loc = node.properties['W']
                            move = self.location_to_move(loc)

                            # store the data
                            states.append(self.board.current_state())
                            probs = np.zeros(self.board.width*self.board.height)
                            probs[[move]] = 1
                            mcts_probs.append(probs)
                            current_players.append(self.board.current_player)
                            # perform a move
                            self.board.do_move(move)
                        else:
                            # winner from the perspective of the current player of each state
                            winners_z = np.zeros(len(current_players))
                            if winner != -1:
                                winners_z[np.array(current_players) == winner] = 1.0
                                winners_z[np.array(current_players) != winner] = -1.0
                            print('winner: ', winner)
                            print('states:', states)
                            print('probs: ', mcts_probs)
                            print('winners_z:')
                            print(winners_z)
                            return winner, zip(states, mcts_probs, winners_z)
        except:
            winner0 = 0
            print('read chess manual fail')
            return winner0, zip(states, mcts_probs, np.zeros(len(current_players)))

    def location_to_move(self, loc):
        x = ord(loc[0][0]) - ord('a')
        y = ord(loc[0][1]) - ord('a')
        return self.board.width * y + x


if __name__ == '__main__':
    list_dirs = os.walk('sgf')
    board = Board(width=15, height=15, n_in_row=5)
    q = Manual(board)
    failed_times = 0
    winners = np.zeros(4)
    for root, dirs, files in list_dirs:
        for file in files:
            file_path = os.path.join(root, file)
            winner, play_data = q.read_manual_data(file_path)
            print(play_data)
            winners[winner] += 1
    print("errors: {}, black wins: {}, white wins: {}, ties: {}".format(winners[0], winners[1], winners[2], winners[-1]))
