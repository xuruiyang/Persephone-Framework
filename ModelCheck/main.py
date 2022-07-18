from Coach import Coach
import copy
from persephone.PersephoneGame import PersephoneGame as Game
from persephone.keras.NNet import NNetWrapper as nn
from utils import *

from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf

args = dotdict({
    'numIters': 1000000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 100000,        #
    'updateThreshold': 0.0,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 5,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    # 'load_folder_file': ('C://Users//ruiya//alpha-zero-general//temp','8_0.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})

if __name__ == "__main__":
    with tf.device("cpu:0"):

        g = Game(6)
        nnet = nn(g)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(g, nnet, args)
        iterations = []
        for _ in range(1):
            g = Game(6)
            c = Coach(g, nnet, args)
            i = c.learn()
            iterations.append(i)
            print(iterations)

