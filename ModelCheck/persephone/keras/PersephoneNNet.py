import sys
sys.path.append('..')
from utils import *

import argparse
import keras as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *

"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class PersephoneNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y
        adv = Input(shape=(1,))
        # adv = Input(shape=(self.action_size,))
        old_prediction = Input(shape=(self.action_size,))

        mlp1 = Dense(128, activation='relu')(self.input_boards)
        mlp2 = Dense(128, activation='relu')(mlp1)

        vmlp1 = Dense(128, activation='relu')(self.input_boards)
        vmlp2 = Dense(128, activation='relu')(vmlp1)
        h_conv4_flat = Flatten()(mlp2)    
        vh_conv4_flat = Flatten()(vmlp2)       
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(128)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(64)(s_fc1))))          # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        # self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1
        vs_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(128)(vh_conv4_flat))))  # batch_size x 1024
        vs_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(64)(vs_fc1))))          # batch_size x 1024
        self.v = Dense(1, activation='tanh', name='v')(vs_fc2)                    # batch_size x 1
        # self.v = Dense(1, activation='relu', name='v')(vs_fc2)                    # batch_size x 1

        def custom_loss(y_target, y):
            out = K.clip(y, 1e-8, 1-1e-8)
            log_like = y_target*K.log(out)
            return K.sum(-log_like*adv)

        def proximal_policy_optimization_loss(advantage, old_prediction):
            def loss(y_true, y_pred):
                LOSS_CLIPPING = 0.2
                ENTROPY_LOSS = 1.0
                prob = K.sum(y_true * y_pred, axis=-1)
                old_prob = K.sum(y_true * old_prediction, axis=-1)
                r = prob/(old_prob + 1e-10)
                # return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS *0* -(prob * K.log(prob + 1e-10)))
                return -K.mean(r * advantage - ENTROPY_LOSS * tf.keras.losses.categorical_crossentropy(old_prediction, y_pred))
            return loss

        # loss_funcs = {
        # "pi" : proximal_policy_optimization_loss(advantage=adv, old_prediction=old_prediction),
        # "v": "mean_squared_error"}

        # loss_funcs = {
        # "pi" : custom_loss,
        # "v": "mean_squared_error"}

        loss_funcs = {
        "pi" : "categorical_crossentropy",
        "v": "mean_squared_error"}

        self.model = Model(inputs=[self.input_boards, adv, old_prediction], outputs=[self.pi, self.v])
        self.model.compile(loss=loss_funcs, optimizer=Adam(args.lr))

        self.predict = Model(inputs=self.input_boards, outputs=[self.pi, self.v])