from Coach import Coach
from persephone.PersephoneGame import PersephoneGame as Game
from persephone.keras.NNet import NNetWrapper as nn
from utils import *

import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf

args = dotdict({
    'numIters': 100000,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 100000,        #
    'updateThreshold': 0.0,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 10,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})

if __name__ == "__main__":
    with tf.device("cpu:0"):
    # if True:
        # 9822
        # 6774
        # 3833
        seed0 = np.random.randint(10000)
        # seed0 = 6144
        print("seed:",seed0)
        seed(seed0)
        set_random_seed(seed0)

        g = Game(6)
        nnet = nn(g, normal_size=True)
        nnet2 = nn(g, normal_size=True)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(g, nnet, nnet2, args)
        if args.load_model:
            print("Load trainExamples from file")
            c.loadTrainExamples()
        c.learn()
