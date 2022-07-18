import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import tensorflow as tf
from .QSATNNet import QSATNNet as onnet

import conf

args = dotdict({
    'lr': 0.0001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 32,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, identity=1):
        self.nnet = onnet(game, args)
        self.action_size = game.getActionSize()
        self.identity=identity

        self.sess = tf.Session(graph=self.nnet.graph)
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(tf.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: list of examples, each example is of form [(init_node_reps,amats,num_vertices,game_state),pi,v]
        """
        flatten = lambda l: [y for x in l for y in (x if isinstance(x, list) else (x,))]

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                init_node_reps,amats,num_vertices,pis,vs = list(zip(*[flatten(examples[i]) for i in sample_ids]))
                
                avs = np.zeros((args.batch_size, 2))
                vs = np.asarray([vs]).T
                vs = np.pad(vs,pad_width=[[0,0],[0,1]],mode='constant')

                labels = [list(pis),vs]
                # predict and compute gradient and do SGD step
                init_node_reps = np.asarray(init_node_reps)
                labels = np.asarray(labels)
                amats = np.asarray(amats)

                input_dict = {
                    self.nnet.placeholders['initial_node_representation']: init_node_reps,
                    self.nnet.placeholders['target_values']: labels,
                    self.nnet.placeholders['num_vertices']: conf.NUM_V,
                    self.nnet.placeholders['adjacency_matrix']: amats,
                    self.nnet.placeholders['graph_state_keep_prob']: 1,
                    self.nnet.placeholders['edge_weight_dropout_keep_prob']: 1,
                    self.nnet.placeholders['out_layer_dropout_keep_prob'] : 1
                }
                
                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                summary, _ = self.sess.run([self.nnet.merged, self.nnet.ops['train_step']], feed_dict=input_dict)
                self.nnet.train_writer.add_summary(summary, batch_idx)
                pi_loss, v_loss, loss, grads = self.sess.run([self.nnet.loss_pi, self.nnet.loss_v, self.nnet.ops['loss'], self.nnet.grads], feed_dict=input_dict)
                pi_losses.update(pi_loss)
                v_losses.update(v_loss)
                losses.update(loss)

                # print(grads)
                # exit()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f} | Loss: {l:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            l=losses.avg
                            )
                bar.next()
            bar.finish()


    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        (init_node_reps,amats,num_vertices,game_state,_) = board

        input_dict = {
                    self.nnet.placeholders['initial_node_representation']: [init_node_reps],
                    self.nnet.placeholders['num_vertices']: max([num_vertices]),
                    self.nnet.placeholders['adjacency_matrix']: [amats],
                    self.nnet.placeholders['graph_state_keep_prob']: 1,
                    self.nnet.placeholders['edge_weight_dropout_keep_prob']: 1,
                    self.nnet.placeholders['out_layer_dropout_keep_prob'] : 1
                    }

        # run
        prob, v, pi, vv, final_rep = self.sess.run([self.nnet.ops['prob'], self.nnet.ops['v'], self.nnet.ops['pi'], self.nnet.ops['vv'], self.nnet.ops['final_node_representations']], feed_dict=input_dict)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        # return prob[0], v[0], pi[0], vv[0], final_rep[0]
        return prob[0], v[0], pi[0], vv[0], 1 if self.identity==1 else -1

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:            
            self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath+'.meta'):
            raise("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)