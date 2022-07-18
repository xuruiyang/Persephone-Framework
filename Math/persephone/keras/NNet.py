import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .PersephoneNNet import PersephoneNNet as onnet

"""
NeuralNet wrapper class for the PersephoneNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.00,
    'epochs': 100,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
    'layer_size': 1024,
})

args2 = dotdict({
    'lr': 0.001,
    'dropout': 0.0,
    'epochs': 20,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
    'layer_size': 256,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, normal_size=True):
        if not normal_size:
            self.nnet = onnet(game, args2)
        else:
            self.nnet = onnet(game, args2)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # input_boards, target_pis, target_vs = list(zip(*examples))
        # input_boards = np.asarray(input_boards)
        # target_pis = np.asarray(target_pis)
        # target_vs = np.asarray(target_vs)
        input_boards, adv, old_prediction, target_pis, target_vs, target_concept = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        old_prediction = np.asarray(old_prediction)
        adv =  np.asarray(adv)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        target_concept = np.asarray(target_concept)

        # train_on_batch ???
        self.nnet.model.fit(x = [input_boards, adv, old_prediction], y = [target_pis, target_vs, target_concept], batch_size = args.batch_size, epochs = args.epochs, verbose=2)
        # self.nnet.model.train_on_batch(x = [input_boards, adv, old_prediction], y = [target_pis, target_vs])

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v, concept = self.nnet.predict.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0], concept[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        # else:
        #     print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
