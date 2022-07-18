import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board,p):
        a = np.random.randint(self.game.getActionSize())
        return a

class Seqlayer():
    def __init__(self, game, seq):
        self.game = game
        self.seq = seq
        self.t = 0

    def play(self, board, p):
        a = self.seq[self.t]
        self.t += 1
        self.t = self.t%10
        return a

class HumanQSATPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, p):
        # display(board)
        a = input()
        return int(a)
